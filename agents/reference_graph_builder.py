"""
引用图构建Agent - 从论文中提取参考文献并构建引用关系图
Reference Graph Builder Agent
"""
import os
import re
import csv
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from openai import OpenAI
from loguru import logger
from difflib import SequenceMatcher

from agents.base_agent import BaseAgent


class ReferenceGraphBuilderAgent(BaseAgent):
    """
    引用图构建Agent

    功能:
    1. 从论文markdown文件中提取References部分
    2. 解析参考文献，提取论文标题和arXiv ID
    3. 在arxiv元数据数据库中搜索匹配的论文
    4. 构建引用关系有向图，输出CSV格式的边列表
    """

    def __init__(
        self,
        client: OpenAI,
        system_prompt: str,
        mcp_tool_manager: Optional[Any] = None
    ):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="ReferenceGraphBuilder"
        )

        # arxiv元数据索引 (title -> arxiv_id)
        self.title_to_id: Dict[str, str] = {}
        self.id_to_title: Dict[str, str] = {}
        # 新增：arxiv ID 到标题的映射（支持多种ID格式）
        self.arxiv_id_to_canonical: Dict[str, str] = {}  # 各种ID格式 -> 标准ID
        self.metadata_loaded = False

    def load_arxiv_metadata(self, metadata_path: str) -> int:
        """
        加载arxiv元数据，构建标题索引

        Args:
            metadata_path: arxiv元数据JSON文件路径

        Returns:
            int: 加载的论文数量
        """
        logger.info(f"正在加载arxiv元数据: {metadata_path}")

        count = 0
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    title = paper.get('title', '').strip().lower()
                    arxiv_id = paper.get('id', '').strip()

                    if title and arxiv_id:
                        # 标准化标题: 移除多余空格和标点
                        normalized_title = self._normalize_title(title)
                        self.title_to_id[normalized_title] = arxiv_id
                        self.id_to_title[arxiv_id] = title
                        count += 1

                        # 同时索引所有版本的ID
                        versions = paper.get('versions', [])
                        for v in versions:
                            version_id = v.get('version', '')
                            if version_id:
                                # 如 v1 -> 0704.1234v1
                                full_version_id = f"{arxiv_id}v{version_id.replace('v', '')}"
                                self.id_to_title[full_version_id] = title

                except json.JSONDecodeError:
                    continue

        self.metadata_loaded = True
        logger.info(f"已加载 {count} 篇论文的元数据索引")
        return count

    def _normalize_title(self, title: str) -> str:
        """
        标准化标题用于匹配

        Args:
            title: 原始标题

        Returns:
            str: 标准化后的标题
        """
        # 转小写
        title = title.lower()
        # 移除多余空格
        title = ' '.join(title.split())
        # 移除常见标点
        title = re.sub(r'[^\w\s]', '', title)
        return title.strip()

    def extract_references_section(self, markdown_content: str) -> Optional[str]:
        """
        从markdown内容中提取References部分

        Args:
            markdown_content: markdown文件内容

        Returns:
            Optional[str]: References部分的文本，如果找不到则返回None
        """
        # 尝试多种可能的References标题格式
        patterns = [
            r'##\s*References?\s*\n(.*?)(?:\n##|\Z)',
            r'#\s*References?\s*\n(.*?)(?:\n#|\Z)',
            r'\nReferences?\s*\n(.*?)(?:\n[A-Z][a-z]+\s*\n|\Z)',
            r'##\s*Bibliography\s*\n(.*?)(?:\n##|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, markdown_content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果上述模式都没匹配到，尝试找最后一个 ## 标题后的内容
        sections = re.split(r'\n##\s+', markdown_content)
        for section in reversed(sections):
            if section.lower().startswith('references') or section.lower().startswith('bibliography'):
                lines = section.split('\n', 1)
                if len(lines) > 1:
                    return lines[1].strip()

        return None

    def parse_references(self, references_text: str) -> List[Dict[str, str]]:
        """
        解析参考文献文本，提取论文信息
        支持学术论文常见的多种引用格式

        Args:
            references_text: References部分的文本

        Returns:
            List[Dict[str, str]]: 解析出的参考文献列表，每个元素包含title, authors, year等
        """
        references = []

        # 检测引用格式类型
        # 格式1: [Author et al., Year] Authors. Title. Venue.
        # 格式2: Authors. Year. Title. Venue. (无标签)
        has_bracket_labels = bool(re.search(r'^\s*\[[^\]]+\]', references_text, re.MULTILINE))

        if has_bracket_labels:
            # 方法1: 按 [xxx] 格式的引用标签分割
            ref_pattern = r'\[([^\]]+)\]\s*'
            parts = re.split(ref_pattern, references_text)

            for i in range(1, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                label = parts[i].strip()
                content = parts[i + 1].strip()
                ref = self._parse_single_reference(label, content)
                if ref and ref.get('title'):
                    references.append(ref)
        else:
            # 方法2: 无标签格式，按换行分割，每行一个引用
            # 格式: Authors. Year. Title. Venue.
            references = self._parse_references_no_labels(references_text)

        # 如果上面的方法没有提取到足够的引用，尝试备用方法
        if len(references) < 3:
            references = self._parse_references_by_line(references_text)

        logger.info(f"从References中解析出 {len(references)} 条参考文献")
        return references

    def _parse_references_no_labels(self, text: str) -> List[Dict[str, str]]:
        """
        解析无标签格式的参考文献
        格式: Authors. Year. Title. Venue. 或 Authors. Title. Year. Venue.
        支持 PDF 转换后引用被连在一起的情况
        """
        references = []

        # 方法1: 使用 arXiv ID 作为分割标志
        # 模式: "arXiv preprint arXiv:xxxx.xxxxx" 标志着一个引用的结束
        arxiv_pattern = r'(arXiv\s*preprint\s*arXiv[:\s]*(\d{4}\.\d{4,5}))'

        # 按 arXiv 标记分割
        parts = re.split(arxiv_pattern, text, flags=re.IGNORECASE)

        # parts 格式: [ref1_text, full_match, arxiv_id, ref2_text, ...]
        # 实际格式: [text_before, 'arXiv preprint arXiv:2301.12345', '2301.12345', text_after, ...]
        i = 0
        while i < len(parts):
            part = parts[i]
            # 检查是否是 arXiv 匹配组
            if i + 2 < len(parts) and re.match(r'^\d{4}\.\d{4,5}$', parts[i+2] if parts[i+2] else ''):
                # parts[i+1] 是完整匹配，parts[i+2] 是 arXiv ID
                arxiv_id = parts[i+2]
                i += 3
                continue
            elif isinstance(part, str) and part.strip():
                # 这是一个引用文本，尝试解析
                ref = self._parse_single_ref_text(part.strip())
                if ref and ref.get('title'):
                    references.append(ref)
                i += 1
            else:
                i += 1

        # 方法2: 如果方法1效果不好，使用作者名模式分割
        if len(references) < 5:
            references = self._parse_by_author_pattern(text)

        return references

    def _parse_by_author_pattern(self, text: str) -> List[Dict[str, str]]:
        """
        通过识别作者名模式来分割引用
        模式: "LastName, FirstName." 或 "FirstName LastName" 后跟年份
        """
        references = []

        # 尝试按句号+空格+大写字母开头的名字分割
        # 这通常标志着一个新引用的开始（作者名）
        # 模式: ". A B C D." 其中 A 是大写姓氏开头

        # 更可靠的方法：查找 "年份. 作者名" 或 "arXiv ID. 作者名" 的模式
        # 分割点模式
        split_patterns = [
            r'(?<=arXiv:\d{4}\.\d{4,5})\.\s+(?=[A-Z][a-z]+)',  # arXiv ID 后
            r'(?<=\d{4})\.\s+(?=[A-Z][a-z]+\s+[A-Z])',  # 年份后跟作者名
            r'(?<=\.)\s+(?=\d{4}\.\s+[A-Z])',  # 句号后跟 "年份. 作者"
        ]

        # 先尝试按 arXiv ID 分割
        arxiv_end_pattern = r'arXiv[:\s]*(\d{4}\.\d{4,5})'
        last_end = 0

        for match in re.finditer(arxiv_end_pattern, text, re.IGNORECASE):
            ref_text = text[last_end:match.end()]
            ref = self._parse_single_ref_text(ref_text.strip())
            if ref and ref.get('title'):
                references.append(ref)
            last_end = match.end()

        # 处理最后一部分
        if last_end < len(text):
            ref_text = text[last_end:].strip()
            if ref_text:
                ref = self._parse_single_ref_text(ref_text)
                if ref and ref.get('title'):
                    references.append(ref)

        return references

    def _parse_single_ref_text(self, text: str) -> Dict[str, str]:
        """
        解析单条参考文献文本（无标签格式）
        格式: Authors. Year. Title. Venue.
        """
        ref = {}
        text = text.strip()

        # 提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            ref['year'] = year_match.group(0)

        # 按句号分割
        parts = re.split(r'\.\s+', text)

        # 标题通常是第一个长句子（>30字符），跳过作者列表
        title_candidate = None
        for i, part in enumerate(parts):
            part = part.strip()
            # 跳过太短的（可能是作者缩写或年份）
            if len(part) < 25:
                continue
            # 跳过看起来像作者列表的
            if self._looks_like_author_list(part):
                continue
            # 跳过期刊/会议名称
            if self._looks_like_venue(part):
                continue

            title_candidate = part
            break

        if title_candidate:
            title_candidate = self._clean_title(title_candidate)
            if len(title_candidate) >= 15:
                ref['title'] = title_candidate

        return ref

    def _parse_single_reference(self, label: str, content: str) -> Dict[str, str]:
        """
        解析单条参考文献

        Args:
            label: 引用标签，如 "Bordes et al., 2013"
            content: 引用内容

        Returns:
            Dict[str, str]: 解析出的引用信息
        """
        ref = {}

        # 从标签中提取年份
        year_match = re.search(r'(\d{4})', label)
        if year_match:
            ref['year'] = year_match.group(1)

        # 从标签中提取第一作者
        author_match = re.match(r'^([A-Z][a-z]+)', label)
        if author_match:
            ref['first_author'] = author_match.group(1)

        # 清理内容
        content = content.strip()

        # 提取标题 - 学术论文引用中，标题通常是第一个以句号结尾的完整句子
        # 且位于作者列表之后

        # 方法1: 查找第一个长句子作为标题（通常30-200字符）
        sentences = re.split(r'\.\s+', content)

        title_candidate = None
        for j, sentence in enumerate(sentences):
            sentence = sentence.strip()
            # 跳过太短的句子（可能是作者名缩写）
            if len(sentence) < 25:
                continue
            # 跳过看起来像作者列表的句子（包含很多大写字母开头的名字）
            if self._looks_like_author_list(sentence):
                continue
            # 跳过期刊/会议名称
            if self._looks_like_venue(sentence):
                continue

            title_candidate = sentence
            break

        if title_candidate:
            # 清理标题
            title_candidate = self._clean_title(title_candidate)
            if len(title_candidate) >= 15:
                ref['title'] = title_candidate

        # 提取作者列表（通常在开头）
        if sentences and not self._looks_like_author_list(sentences[0]):
            # 第一个短句可能是标题，作者可能在前面
            pass
        else:
            # 尝试从内容开头提取作者
            author_part = content.split('.')[0] if '.' in content else content[:100]
            ref['authors_raw'] = author_part.strip()

        return ref

    def _looks_like_author_list(self, text: str) -> bool:
        """判断文本是否像作者列表"""
        # 作者列表特征：多个 "Name, N." 或 "Name and Name" 格式
        if re.search(r'[A-Z][a-z]+,\s*[A-Z]\.', text):
            return True
        if re.search(r'and\s+[A-Z][a-z]+\s+[A-Z]', text):
            return True
        # 多个以大写开头的单词，可能是作者列表
        words = text.split()
        capitalized = sum(1 for w in words[:10] if w and w[0].isupper())
        if capitalized >= 4 and len(text) < 150:
            return True
        return False

    def _looks_like_venue(self, text: str) -> bool:
        """判断文本是否像期刊/会议名称"""
        venue_keywords = [
            'proceedings', 'journal', 'conference', 'ieee', 'acm',
            'arxiv', 'preprint', 'vol.', 'volume', 'pages', 'pp.',
            'springer', 'elsevier', 'nature', 'science', 'press',
            'transactions', 'letters', 'review', 'workshop'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in venue_keywords)

    def _clean_title(self, title: str) -> str:
        """清理标题"""
        # 移除开头的数字和标点
        title = re.sub(r'^[\d\.\[\]\-\s]+', '', title)
        # 移除 LaTeX 特殊字符
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'[\\{}]', '', title)
        # 标准化空格
        title = ' '.join(title.split())
        return title.strip()

    def _parse_references_by_line(self, text: str) -> List[Dict[str, str]]:
        """
        按行解析参考文献（备用方法）
        """
        references = []
        lines = text.split('\n')

        current_ref = {}
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测是否是新引用的开始
            is_new_ref = bool(re.match(r'^\[[^\]]+\]', line)) or \
                         bool(re.match(r'^\d+\.\s+', line))

            if is_new_ref and current_text:
                # 保存当前引用
                ref = self._extract_ref_from_text(' '.join(current_text))
                if ref.get('title'):
                    references.append(ref)
                current_text = []

            current_text.append(line)

        # 处理最后一个引用
        if current_text:
            ref = self._extract_ref_from_text(' '.join(current_text))
            if ref.get('title'):
                references.append(ref)

        return references

    def _extract_ref_from_text(self, text: str) -> Dict[str, str]:
        """从文本中提取引用信息"""
        ref = {}

        # 提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            ref['year'] = year_match.group(0)

        # 按句号分割，找到标题
        sentences = re.split(r'\.\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            if self._looks_like_author_list(sentence):
                continue
            if self._looks_like_venue(sentence):
                continue

            # 清理并作为标题
            title = self._clean_title(sentence)
            if len(title) >= 15:
                ref['title'] = title
                break

        return ref

    def find_matching_paper(self, ref_title: str, ref_author: str = '', threshold: float = 0.75) -> Optional[str]:
        """
        在arxiv元数据中查找匹配的论文

        Args:
            ref_title: 参考文献标题
            ref_author: 参考文献作者（可选，用于辅助匹配）
            threshold: 相似度阈值（降低到0.75以提高匹配率）

        Returns:
            Optional[str]: 匹配的arxiv_id，如果没找到则返回None
        """
        if not self.metadata_loaded:
            logger.warning("元数据未加载，无法进行匹配")
            return None

        normalized_ref = self._normalize_title(ref_title)

        if len(normalized_ref) < 10:
            return None

        # 1. 精确匹配
        if normalized_ref in self.title_to_id:
            return self.title_to_id[normalized_ref]

        # 2. 部分匹配 - 使用词集合来提高匹配效率
        ref_words = set(normalized_ref.split())
        if len(ref_words) < 3:
            return None

        best_match = None
        best_score = 0

        # 为了提高效率，只对包含至少一个关键词的标题进行匹配
        # 先筛选候选
        candidates = []
        for title, arxiv_id in self.title_to_id.items():
            # 快速筛选：长度相近
            if abs(len(title) - len(normalized_ref)) > len(normalized_ref) * 0.5:
                continue
            # 快速筛选：至少有2个词相同
            title_words = set(title.split())
            common_words = ref_words & title_words
            if len(common_words) >= 2:
                candidates.append((title, arxiv_id, len(common_words)))

        # 按共同词数量排序，优先匹配相似度高的
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 只检查前500个候选
        for title, arxiv_id, _ in candidates[:500]:
            score = SequenceMatcher(None, normalized_ref, title).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = arxiv_id
                # 如果找到高置信度匹配，提前返回
                if score >= 0.9:
                    return best_match

        return best_match

    def build_reference_graph(
        self,
        paper_data: List[Dict[str, Any]],
        output_csv_path: str
    ) -> Dict[str, Any]:
        """
        构建引用关系图

        Args:
            paper_data: 论文数据列表，每个元素包含paper_id, markdown_path等
            output_csv_path: 输出CSV文件路径

        Returns:
            Dict[str, Any]: 构建结果统计
        """
        edges = []
        stats = {
            'total_papers': len(paper_data),
            'papers_with_references': 0,
            'total_references_extracted': 0,
            'matched_references': 0,
            'edges_created': 0
        }

        for paper_info in paper_data:
            paper_id = paper_info.get('paper_id', '')
            markdown_path = paper_info.get('markdown_path', '')

            if not os.path.exists(markdown_path):
                logger.warning(f"Markdown文件不存在: {markdown_path}")
                continue

            # 读取markdown内容
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取References部分
            references_text = self.extract_references_section(content)
            if not references_text:
                logger.warning(f"未找到References部分: {paper_id}")
                continue

            stats['papers_with_references'] += 1

            # 解析参考文献
            references = self.parse_references(references_text)
            stats['total_references_extracted'] += len(references)

            # 匹配并创建边
            for ref in references:
                ref_title = ref.get('title', '')
                if not ref_title:
                    continue

                matched_id = self.find_matching_paper(ref_title)
                if matched_id:
                    stats['matched_references'] += 1

                    # 创建边: paper_id -> matched_id (有向边)
                    edge = {
                        'source_id': paper_id,
                        'target_id': matched_id,
                        'source_title': paper_info.get('title', ''),
                        'target_title': self.id_to_title.get(matched_id, ''),
                        'reference_text': ref_title[:200]  # 截断以防过长
                    }
                    edges.append(edge)
                    stats['edges_created'] += 1

        # 保存边到CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source_id', 'target_id', 'source_title', 'target_title', 'reference_text'])
            writer.writeheader()
            writer.writerows(edges)

        logger.info(f"引用图已保存到: {output_csv_path}")
        logger.info(f"统计: {stats}")

        return {
            'status': 'success',
            'edges_file': output_csv_path,
            'edge_count': len(edges),
            'stats': stats
        }

    def execute(self, task: Dict[str, Any], upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行引用图构建任务

        Args:
            task: 任务配置，包含:
                - metadata_path: arxiv元数据JSON文件路径
                - output_dir: 输出目录
            upstream_data: 上游Agent传递的数据，包含论文列表

        Returns:
            Dict[str, Any]: 执行结果
        """
        logger.info(f"[{self.agent_name}] 开始执行引用图构建任务")

        try:
            # 获取配置
            metadata_path = task.get('metadata_path', '/home/QuanWei/arxiv-metadata-cs.json')
            output_dir = task.get('output_dir', './data/graphs')
            output_csv = os.path.join(output_dir, 'citation_edges.csv')

            # 加载arxiv元数据
            if not self.metadata_loaded:
                self.load_arxiv_metadata(metadata_path)

            # 从上游数据获取论文列表
            papers = []
            if upstream_data:
                # 从上游Agent的输出中提取论文信息
                if 'papers' in upstream_data:
                    for paper in upstream_data['papers']:
                        papers.append({
                            'paper_id': paper.get('arxiv_id', paper.get('id', '')),
                            'title': paper.get('title', ''),
                            'markdown_path': paper.get('markdown_path', '')
                        })
                elif 'cleaned_data' in upstream_data:
                    for item in upstream_data['cleaned_data']:
                        papers.append({
                            'paper_id': item.get('paper_id', ''),
                            'title': item.get('title', ''),
                            'markdown_path': item.get('markdown_path', '')
                        })

            if not papers:
                return {
                    'status': 'error',
                    'message': '没有找到论文数据，请确保上游Agent已正确执行'
                }

            # 构建引用图
            result = self.build_reference_graph(papers, output_csv)

            return {
                'status': 'success',
                'edges_file': result['edges_file'],
                'edge_count': result['edge_count'],
                'stats': result['stats'],
                'papers_processed': len(papers)
            }

        except Exception as e:
            logger.error(f"[{self.agent_name}] 执行失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

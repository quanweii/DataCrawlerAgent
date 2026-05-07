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
from multiprocessing import Pool

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from agents.base_agent import BaseAgent

# ==================== 多进程共享状态 ====================
# 在主进程中加载元数据后赋值，子进程通过fork继承（COW，零拷贝）
_shared_title_to_id: Dict[str, str] = {}
_shared_id_to_title: Dict[str, str] = {}
_shared_arxiv_id_to_canonical: Dict[str, str] = {}
_shared_word_index: Dict[str, Set[str]] = {}


def _process_single_paper_refs(paper_info: Dict) -> List[Dict]:
    """
    多进程worker: 处理单篇论文的引用提取和匹配
    使用模块级别的共享元数据索引（通过fork COW继承）
    """
    agent = ReferenceGraphBuilderAgent(client=None, system_prompt="")
    # 指向共享数据，不重新加载
    agent.title_to_id = _shared_title_to_id
    agent.id_to_title = _shared_id_to_title
    agent.arxiv_id_to_canonical = _shared_arxiv_id_to_canonical
    agent._word_index = _shared_word_index
    agent.metadata_loaded = True

    paper_id = paper_info.get('paper_id', '')
    markdown_path = paper_info.get('markdown_path', '')
    pdf_path = paper_info.get('pdf_path', '')

    edges = []
    all_candidates: List[Dict] = []

    # 来源1: pdfplumber
    if pdf_path and os.path.exists(pdf_path):
        try:
            pdf_text = agent.extract_references_from_pdf(pdf_path)
            if pdf_text:
                pdf_refs = agent.parse_references(pdf_text)
                all_candidates.extend(pdf_refs)
        except Exception:
            pass

    # 来源2: markdown
    if os.path.exists(markdown_path):
        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            md_text = agent.extract_references_section(content)
            if md_text:
                md_refs = agent.parse_references(md_text)
                all_candidates.extend(md_refs)
        except Exception:
            pass

    if not all_candidates:
        return []

    # 去重
    seen_titles = set()
    unique_refs = []
    for ref in all_candidates:
        title = ref.get('title', '')
        if not title:
            continue
        normalized = agent._normalize_title(title)
        if normalized and normalized not in seen_titles:
            seen_titles.add(normalized)
            unique_refs.append(ref)
    references = unique_refs

    # 匹配
    for ref in references:
        ref_title = ref.get('title', '')
        ref_arxiv_id = ref.get('arxiv_id', '')

        title_is_garbled = agent._is_garbled_title(ref_title) if ref_title else True
        if not ref_title or title_is_garbled:
            if not ref_arxiv_id:
                continue

        matched_id = None
        if ref_title and not title_is_garbled:
            matched_id = agent.find_matching_paper(ref_title, ref_arxiv_id='')

        if not matched_id and ref_arxiv_id:
            arxiv_match_id = agent.find_matching_paper(ref_title, ref_arxiv_id=ref_arxiv_id)
            if arxiv_match_id:
                raw_text = ref.get('_raw_text', ref_title)
                matched_title = agent.id_to_title.get(arxiv_match_id, '')
                if matched_title and raw_text:
                    if not agent._validate_keyword_overlap(
                            agent._normalize_title(raw_text),
                            agent._normalize_title(matched_title), 0.40):
                        continue
                matched_id = arxiv_match_id

        if matched_id:
            edges.append({
                'source_id': paper_id,
                'target_id': matched_id,
                'source_title': paper_info.get('title', ''),
                'target_title': agent.id_to_title.get(matched_id, ''),
                'reference_text': ref_title[:200]
            })

    return edges


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
        # 倒排索引：关键词 -> set(normalized_titles) 用于快速查找候选
        self._word_index: Dict[str, Set[str]] = {}
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

                        # 构建arxiv ID索引：各种格式 → 标准ID（参考zotero-reference的identifier匹配）
                        self.arxiv_id_to_canonical[arxiv_id] = arxiv_id
                        base_id = arxiv_id.split('v')[0]
                        if base_id != arxiv_id:
                            self.arxiv_id_to_canonical[base_id] = arxiv_id

                        count += 1

                        # 同时索引所有版本的ID
                        versions = paper.get('versions', [])
                        for v in versions:
                            version_id = v.get('version', '')
                            if version_id:
                                full_version_id = f"{arxiv_id}v{version_id.replace('v', '')}"
                                self.id_to_title[full_version_id] = title
                                self.arxiv_id_to_canonical[full_version_id] = arxiv_id

                except json.JSONDecodeError:
                    continue

        # 构建倒排索引：每个关键词 -> 包含该词的标题集合
        logger.info("正在构建关键词倒排索引...")
        stop_words = {'a', 'an', 'the', 'of', 'in', 'for', 'and', 'on', 'to', 'with',
                      'by', 'at', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                      'as', 'or', 'not', 'no', 'but', 'its', 'it', 'this', 'that',
                      'we', 'our', 'their', 'can', 'will', 'all', 'using', 'based'}
        for title in self.title_to_id:
            words = set(title.split()) - stop_words
            for w in words:
                if len(w) >= 3:
                    if w not in self._word_index:
                        self._word_index[w] = set()
                    self._word_index[w].add(title)

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
        # 修复OCR断字: "learn- ing" → "learning"
        title = re.sub(r'(\w)-\s+([a-z])', r'\1\2', title)
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
        # 收集所有可能的References区域
        candidates = []

        # 模式1: ## References 标题后的内容
        for match in re.finditer(r'##\s*References?\s*\n(.*?)(?:\n##|\Z)', markdown_content, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1).strip())

        # 模式2: # References 标题后的内容
        for match in re.finditer(r'#\s*References?\s*\n(.*?)(?:\n#|\Z)', markdown_content, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1).strip())

        # 模式3: Bibliography
        for match in re.finditer(r'##\s*Bibliography\s*\n(.*?)(?:\n##|\Z)', markdown_content, re.DOTALL | re.IGNORECASE):
            candidates.append(match.group(1).strip())

        # 模式4: "References" 行（无#前缀）后跟引用内容
        # 这种情况是OCR把References标题放在了正文中
        for match in re.finditer(r'\nReferences?\s*\n((?:[^\n]*\n)*?)(?=\n##|\Z)', markdown_content, re.DOTALL | re.IGNORECASE):
            content = match.group(1).strip()
            if content:
                candidates.append(content)

        # 模式5: "References Author," 格式（引用紧跟在References后）
        for match in re.finditer(r'\nReferences\s+([A-Z][^\n]*(?:\n(?![##])[^\n]+)*)', markdown_content, re.DOTALL):
            candidates.append(match.group(1).strip())

        # 对候选区域评分，选择引用最多的
        best_content = None
        best_score = -1

        for content in candidates:
            # 计算引用指标
            bracket_refs = len(re.findall(r'\[[^\]]+?,\s*\d{4}\]', content))
            number_bracket_refs = len(re.findall(r'\[\d{1,3}\]', content))
            arxiv_refs = len(re.findall(r'arXiv[:\s]*\d{4}\.\d{4,5}', content, re.IGNORECASE))
            arxiv_url_refs = len(re.findall(r'arxiv\.0?rg/abs/\d{4}\.\d{4,5}', content, re.IGNORECASE))
            year_refs = len(re.findall(r'\b(19|20)\d{2}\b', content))
            lines_with_content = sum(1 for line in content.split('\n') if len(line.strip()) > 20)

            score = (bracket_refs * 3 + number_bracket_refs * 2 +
                     arxiv_refs * 5 + arxiv_url_refs * 4 +
                     year_refs + lines_with_content)

            if score > best_score:
                best_score = score
                best_content = content

        # 如果找到了内容但引用很少，尝试其他方法
        if best_content and best_score < 5:
            # 查找文档末尾的引用密集区域
            lines = markdown_content.split('\n')
            ref_lines = []
            in_ref_area = False

            for i, line in enumerate(lines):
                # 检测引用密集区域
                if re.search(r'arXiv[:\s]*\d{4}\.\d{4,5}', line, re.IGNORECASE) or \
                   re.search(r'arxiv\.0?rg/abs/\d{4}\.\d{4,5}', line, re.IGNORECASE) or \
                   re.search(r'^[A-Z][a-z]+.*,\s*\d{4}[a-z]?\.\s', line):
                    in_ref_area = True
                    ref_lines.append(line)
                elif in_ref_area and line.strip():
                    ref_lines.append(line)
                elif in_ref_area and not line.strip():
                    # 空行可能表示引用区域结束
                    if ref_lines and len(ref_lines) > 5:
                        in_ref_area = False

            if len(ref_lines) > len(best_content.split('\n')):
                best_content = '\n'.join(ref_lines)

        # 最终回退: 搜索包含密集[N]编号的区域（即使没有References标题）
        if not best_content or best_score < 3:
            lines = markdown_content.split('\n')
            # 找到[N]模式出现次数最多的区域（滑动窗口）
            bracket_line_indices = []
            for i, line in enumerate(lines):
                count = len(re.findall(r'\[\d{1,3}\]', line))
                if count >= 2:
                    bracket_line_indices.append((i, count))

            if len(bracket_line_indices) >= 2:
                # 找到最密集的连续区域
                first_idx = bracket_line_indices[0][0]
                last_idx = bracket_line_indices[-1][0]
                # 不要太早开始（避免正文中偶尔的引用标记）
                start = max(0, first_idx - 1)
                end = min(len(lines), last_idx + 2)
                candidate = '\n'.join(lines[start:end])
                # 验证确实有足够的引用编号
                bracket_count = len(re.findall(r'\[\d{1,3}\]', candidate))
                if bracket_count >= 5:
                    bracket_score = bracket_count * 3 + len(re.findall(r'\b(19|20)\d{2}\b', candidate))
                    if bracket_score > best_score:
                        best_content = candidate

        return best_content if best_content else None

    def _split_concatenated_refs(self, text: str) -> str:
        """
        预处理：将MD中连在一起的引用行拆分为独立行
        pdf-extract-kit常将多条引用输出为同一行，如:
          "1. Author. Title. 2. Author. Title. 3. Author. Title."
          "[1] Author. Title. [2] Author. Title."
        或:
          "Author (2023). Title. Author2 (2021). Title2."
        """
        lines = text.split('\n')
        result_lines = []

        for line in lines:
            # 只处理超长行（>200字符且不是单条完整引用）
            if len(line) < 200:
                result_lines.append(line)
                continue

            # 策略0: 按 [N] 方括号编号模式拆分（同一行内的多条引用）
            # 匹配: "[1] Author..." "[2] Author..." 等
            bracket_positions = [m.start() for m in re.finditer(
                r'\[\d{1,3}\]\s*[A-Z(]', line
            )]
            if len(bracket_positions) >= 3:
                parts = []
                for i, pos in enumerate(bracket_positions):
                    end = bracket_positions[i + 1] if i + 1 < len(bracket_positions) else len(line)
                    parts.append(line[pos:end].strip())
                result_lines.extend(parts)
                continue

            # 策略1: 按 "N. " 或 "N) " 编号模式拆分
            # 匹配: "2. Author" 或 "2) Author"（也处理全角句号．）
            # 排除 4位数字年份 如 "2024. Author"
            number_positions = [m.start() for m in re.finditer(
                r'(?<![0-9])(?:\d{1,3}[.．\)]\s+[A-Z])', line
            )]
            if len(number_positions) >= 3:
                parts = []
                for i, pos in enumerate(number_positions):
                    end = number_positions[i + 1] if i + 1 < len(number_positions) else len(line)
                    parts.append(line[pos:end].strip())
                result_lines.extend(parts)
                continue

            # 策略2: 按作者名+年份模式拆分
            author_year_split = re.split(
                r'(?<=[\.\"])\s+(?=[A-Z][a-z]+(?:,\s*[A-Z]\.)*(?:\s+(?:and|&))?\s*(?:\(|[A-Z]))',
                line
            )
            if len(author_year_split) > 3:
                result_lines.extend(author_year_split)
                continue

            # 策略3: 按 arXiv ID 边界拆分（同时匹配 arXiv: 和 arxiv.org/abs/ 格式）
            arxiv_positions = [m.end() for m in re.finditer(
                r'(?:arXiv[:\s]*\d{4}\.\d{4,5}|arxiv\.0?rg/abs/\d{4}\.\d{4,5})',
                line, re.IGNORECASE
            )]
            if len(arxiv_positions) >= 2:
                parts = []
                last = 0
                for pos in arxiv_positions:
                    if pos > last:
                        parts.append(line[last:pos])
                    last = pos
                if last < len(line):
                    parts.append(line[last:])
                if len(parts) > 2:
                    result_lines.extend(parts)
                    continue

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _split_concatenated_refs_v2(self, text: str) -> str:
        """
        v2: 多信号边界检测分割引用行
        改进: 无200字符阈值、统一评分、弹性N.前看
        """
        lines = text.split('\n')
        result_lines = []

        for line in lines:
            boundaries = self._find_split_boundaries(line)

            # 至少2个有效边界才分割
            valid = [b for b in boundaries if b[1] >= 3]
            if len(valid) < 2:
                result_lines.append(line)
                continue

            # 合并距离<10的边界（保留高分）
            merged = []
            for pos, score in valid:
                if merged and pos - merged[-1][0] < 10:
                    if score > merged[-1][1]:
                        merged[-1] = (pos, score)
                else:
                    merged.append((pos, score))

            if len(merged) < 2:
                result_lines.append(line)
                continue

            # 在边界位置分割
            positions = [p for p, _ in merged]
            parts = []
            prev = 0
            for pos in positions:
                parts.append(line[prev:pos].strip())
                prev = pos
            parts.append(line[prev:].strip())
            result_lines.extend(p for p in parts if p)

        return '\n'.join(result_lines)

    def _find_split_boundaries(self, line: str) -> List[Tuple[int, int]]:
        """
        找到所有候选分割边界，返回 (position, score) 列表
        """
        boundaries = []

        # 信号1: [N] 方括号编号 — 分数 10
        for m in re.finditer(r'\[\d{1,3}\]\s*', line):
            boundaries.append((m.start(), 10))

        # 信号2: N. / N) 编号 — 弹性前看，跳过 $...$ LaTeX
        for m in re.finditer(r'(?<![0-9])(?:\d{1,3}[.．\)]\s+)', line):
            score = 8
            # 看匹配之后的文本，跳过 LaTeX
            ahead = line[m.end():m.end() + 50]
            ahead_clean = re.sub(r'\$[^$]*\$', '', ahead).strip()
            if ahead_clean and ahead_clean[0].isupper():
                score = 10
            elif ahead_clean and ahead_clean[0].isdigit():
                score = 4  # 可能是年份，降低置信度
            boundaries.append((m.start(), score))

        # 信号3: 年份结束边界 — "2023. " 后跟大写字母（新引用开始）
        for m in re.finditer(r'\b(?:19|20)\d{2}[a-z]?\.\s+(?=[A-Z])', line):
            boundaries.append((m.start(), 3))

        # 信号4: arXiv ID + 年份结束后的新引用边界
        for m in re.finditer(
            r'(?:arXiv[:\s]*\d{4}\.\d{4,5}|arxiv\.0?rg/abs/\d{4}\.\d{4,5})'
            r'[,，]?\s*(?:19|20)\d{2}[a-z]?\.\s+(?=[A-Z\d])',
            line, re.IGNORECASE
        ):
            boundaries.append((m.start(), 5))

        return sorted(boundaries, key=lambda b: b[0])

    def _post_merge_fragments(self, lines: List[str]) -> List[str]:
        """
        后处理：检测并合并碎片（裸arXiv、出版商名、年份尾部等）
        """
        if not lines:
            return lines

        result = []
        for line in lines:
            kind = self._classify_fragment(line)

            if kind == 'complete':
                result.append(line)
            else:
                # 碎片 — 合并到前一条
                if result:
                    merged_len = len(result[-1]) + len(line) + 1
                    if merged_len <= 2000:
                        # OCR 修复后合并
                        cleaned = self._clean_ocr_artifacts(line)
                        result[-1] = result[-1] + ' ' + cleaned
                    else:
                        result.append(line)
                else:
                    # 首条就是碎片：有 arXiv ID 就保留
                    if re.search(r'\d{4}\.\d{4,5}', line):
                        result.append(line)
                    # 否则丢弃

        return result

    def _classify_fragment(self, text: str) -> str:
        """分类一条切分结果：是完整引用还是需要合并的碎片"""
        text = text.strip()
        if not text:
            return 'empty'

        # 裸 arXiv: 短文本主要是 arXiv 标识
        if len(text) < 80 and re.match(
            r'^[\d.．\s]*arXiv\s*(preprint\s*)?(arXiv:)?:?\s*\d{4}\.\d{4,5}',
            text, re.IGNORECASE
        ):
            return 'arxiv_tail'

        # 裸 DOI
        if len(text) < 80 and re.match(
            r'^[\d.．\s]*(DOI:|doi:|10\.\d{4,9}/)', text, re.IGNORECASE
        ):
            return 'doi_tail'

        # 出版商/会议名尾部
        if len(text) < 60 and re.match(
            r'^[\d.．\s]*(Springer|IEEE|ACM|Elsevier|Wiley|Oxford|Cambridge|'
            r'PMLR|Curran|MIT Press|pp\.\s*\d|vol\.\s*\d)',
            text, re.IGNORECASE
        ):
            return 'venue_tail'

        # 纯年份
        if len(text) < 20 and re.match(
            r'^[\d.．\s]*\(?(19|20)\d{2}\)?[.,;]?\s*$', text
        ):
            return 'year_tail'

        # 裸 URL
        if len(text) < 120 and re.match(
            r'^[\d.．\s]*https?://', text, re.IGNORECASE
        ):
            return 'url_tail'

        # OCR 碎片：单词粘连 arXiv ID
        if re.search(r'[a-z]arXiv[:\s]*\d{4}\.\d{4,5}', text, re.IGNORECASE):
            return 'ocr_fragment'
        # OCR 碎片：长无空格 CamelCase
        if re.search(r'[a-z]{15,}', text) and len(text) < 100:
            return 'ocr_fragment'

        return 'complete'

    def _clean_ocr_artifacts(self, text: str) -> str:
        """修复 OCR 导致的文本粘连"""
        # 单词粘连 arXiv: "notationsarXiv:XXXX" → "notations. arXiv:XXXX"
        text = re.sub(
            r'([a-z])(arXiv[:\s]*\d{4}\.\d{4,5})',
            r'\1. \2', text, flags=re.IGNORECASE)
        # 单词粘连年份: "in2020," → "in 2020,"
        text = re.sub(r'([a-z])((?:19|20)\d{2})\b', r'\1 \2', text)
        # 出版社粘连年份: "Springer(2024)" → "Springer (2024)"
        text = re.sub(r'([a-zA-Z])\((\d{4})', r'\1 (\2', text)
        return text

    def parse_references(self, references_text: str) -> List[Dict[str, str]]:
        """
        解析参考文献文本，提取论文信息
        支持学术论文常见的多种引用格式

        Args:
            references_text: References部分的文本

        Returns:
            List[Dict[str, str]]: 解析出的参考文献列表，每个元素包含title, authors, year等
        """
        # 预处理：拆分MD中连在一起的引用行
        references_text = self._split_concatenated_refs(references_text)
        # 后处理：合并碎片（裸arXiv、出版商名、年份尾部）
        lines = [l for l in references_text.split('\n') if l.strip()]
        lines = self._post_merge_fragments(lines)
        references_text = '\n'.join(lines)

        references = []

        # 检测引用格式类型（参考zotero-reference的8种引用格式检测）
        # 格式1: [Author et al., Year] Authors. Title. Venue.
        # 格式2: [数字] Authors. Title. Venue.
        # 检查行首或行内的[N]模式
        has_bracket_labels = bool(re.search(r'^\s*\[[^\]]+\]', references_text, re.MULTILINE))
        # 如果行首没有，检查行内是否有密集的[N]模式
        if not has_bracket_labels:
            inline_brackets = re.findall(r'\[\d{1,3}\]', references_text)
            if len(inline_brackets) >= 3:
                has_bracket_labels = True

        # 格式3: (数字) Authors. Title. Venue.
        has_paren_number = bool(re.search(r'^\s*\(\d+\)\s?', references_text, re.MULTILINE))

        # 格式4: ［数字］（全角括号）
        has_fullwidth_bracket = bool(re.search(r'^\s*［\d+］', references_text, re.MULTILINE))

        # 格式5: 数字编号 1. Author: Title.
        has_number_labels = bool(re.search(r'^\s*\d+\.\s+[A-Z]', references_text, re.MULTILINE))

        # 格式6: 数字 空格 Author（无句号）
        has_number_space = bool(re.search(r'^\s*\d+\s+[A-Z][a-z]+', references_text, re.MULTILINE))

        # 格式7: [RCK+20] 风格的作者-年代编码
        has_author_code = bool(re.search(r'^\s*\[[A-Z]+[-+]?[0-9]{2}\]', references_text, re.MULTILINE))

        if has_bracket_labels or has_author_code:
            # 按 [xxx] 格式的引用标签分割（包括 [1], [Author+20] 等）
            ref_pattern = r'\[[^\]]*\]\s*'
            parts = re.split(ref_pattern, references_text)

            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)

        elif has_paren_number:
            # (数字) 格式: (1) Authors. Title. Venue.
            parts = re.split(r'\n\s*\(\d+\)\s*', references_text)
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)

        elif has_fullwidth_bracket:
            # ［数字］全角括号格式
            parts = re.split(r'\n\s*［\d+］\s*', references_text)
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)

        elif has_number_labels:
            # 按数字编号分割: "1. Author..." / "2. Author..."
            parts = re.split(r'\n\s*\d+\.\s+', references_text)
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)

        elif has_number_space:
            # 数字+空格格式: "1 Author..." (无句号)
            parts = re.split(r'\n\s*\d+\s+(?=[A-Z])', references_text)
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)

        else:
            # 无标签格式
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

        # 方法1: 按行解析（如果文本有换行，每行可能是一条引用）
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) > 3:
            for line in lines:
                ref = self._parse_single_ref_text(line)
                if ref and ref.get('title'):
                    references.append(ref)
            if len(references) >= 3:
                return references

        # 方法2: 使用引用结束标记分割（年份后跟作者名模式）
        # 常见模式: "... 2023. Author, F., ..." 或 "... arXiv:xxxx.xxxxx. Author, ..."
        # 用finditer找到分割点，避免可变宽度look-behind
        parts = []
        split_points = [m.start() for m in re.finditer(
            r'(?:20|19)\d{2}[a-z]?\.\s+(?=[A-Z][a-z]+(?:,\s*[A-Z]\.)+)', text
        )]
        if len(split_points) >= 2:
            last = 0
            for pos in split_points:
                dot_pos = text.rfind('.', last, pos + 4)
                if dot_pos > last:
                    parts.append(text[last:dot_pos + 1])
                    last = dot_pos + 1
            if last < len(text):
                parts.append(text[last:])
        if len(parts) > 2:
            references = []
            for part in parts:
                part = part.strip()
                if not part or len(part) < 20:
                    continue
                ref = self._parse_single_ref_text(part)
                if ref and ref.get('title'):
                    references.append(ref)
            if len(references) >= 3:
                return references

        # 方法3: 使用 arXiv ID 作为分割标志
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

    def _compute_title_score(self, segment: str) -> float:
        """
        计算片段作为标题的可能性得分（越低越可能是标题）
        参考zotero-reference的abbreviation density方法:
        统计缩写[A-Z].、标点、数字的出现密度，密度越低越可能是标题
        """
        if len(segment) < 15:
            return float('inf')

        # 缩写模式 (J., K., etc.)
        abbrev_count = len(re.findall(r'\b[A-Z]\.', segment))
        # 标点密度 (逗号、分号多 = 可能是作者列表)
        punct_count = len(re.findall(r'[,;:]', segment))
        # 数字密度 (年份、卷号 = 可能是期刊)
        digit_count = len(re.findall(r'\d', segment))
        # 作者名模式 "Name, N."
        author_pattern = len(re.findall(r'[A-Z][a-z]+,\s*[A-Z]\.', segment))
        # 空格数（zotero-reference: 过滤少于3个空格的=期刊缩写）
        space_count = len(re.findall(r'\s+', segment))

        # 作者名特征惩罚（区分作者列表和标题）
        author_like = 0
        # "FirstName ... and FirstName ..." 模式
        if re.search(r'[A-Z][a-z]+\s+and\s+[A-Z][a-z]+', segment):
            author_like += 3
        # "Name, N." 模式
        if re.search(r'[A-Z][a-z]+,\s*[A-Z]\.', segment):
            author_like += 2
        # "et al." 模式
        if re.search(r'\bet\s+al\.?', segment, re.IGNORECASE):
            author_like += 4
        # 几乎所有单词首字母大写 = 专有名词（人名），不是标题
        words = segment.split()
        if len(words) >= 3:
            cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
            if cap_ratio > 0.85:
                author_like += 1.5
        # OCR 无空格检测：长文本极少空格
        if space_count <= 2 and len(segment) > 30:
            author_like += 3

        total_len = len(segment)
        # 综合密度得分
        density = (abbrev_count * 2 + punct_count * 1.5 + digit_count * 0.5 +
                   author_pattern * 3 + author_like * 2) / max(total_len, 1)
        # 空格惩罚：空格太少（紧凑格式）不太可能是标题
        if space_count < 3:
            density += 0.3

        return density

    def _parse_single_ref_text(self, text: str) -> Dict[str, str]:
        """
        解析单条参考文献文本（无标签格式）
        支持: Authors. Year. Title. Venue. / Authors: Title. Venue, Year. / Authors (Year) Title
        """
        ref = {}
        text = text.strip()
        # 保存原始文本用于后续arXiv URL回退提取
        ref['_raw_text'] = text

        # 提取年份
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            ref['year'] = year_match.group(0)

        # 提取arXiv ID（多种格式）
        # 格式1: arXiv:NNNN.NNNNN 或 arXiv preprint arXiv:NNNN.NNNNN
        arxiv_match = re.search(r'arXiv[:\s]*(\d{4}\.\d{4,5})', text, re.IGNORECASE)
        if not arxiv_match:
            # 格式2: arxiv.org/abs/NNNN.NNNNN（处理OCR损坏如 arxiv.0rg, https:/ /）
            arxiv_match = re.search(r'arxiv\.0?rg/abs/(\d{4}\.\d{4,5})', text, re.IGNORECASE)
        if not arxiv_match:
            # 格式3: https://arxiv.org/pdf/NNNN.NNNNN
            arxiv_match = re.search(r'arxiv\.0?rg/(?:abs|pdf)/(\d{4}\.\d{4,5})', text, re.IGNORECASE)
        if arxiv_match:
            ref['arxiv_id'] = arxiv_match.group(1)

        # 提取DOI（参考zotero-reference的identifier提取）
        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+', text)
        if doi_match:
            ref['doi'] = doi_match.group(0).rstrip('.')

        # 尝试多种标题提取策略

        # 策略1: 查找带引号的标题
        quoted = re.search(r'["\u201c\u201e]([^"\u201d]{15,}?)["\u201d]', text)
        if quoted:
            title = self._clean_title(quoted.group(1))
            if len(title) >= 15:
                ref['title'] = title
                return ref

        # 策略2: 按句号分割，用缩写密度评分选择标题（参考zotero-reference）
        parts = re.split(r'\.\s+', text)
        scored_parts = []
        for part in parts:
            part = part.strip()
            if len(part) < 15:
                continue
            score = self._compute_title_score(part)
            scored_parts.append((part, score))

        if scored_parts:
            # 选择缩写密度最低的片段作为标题（最不像作者/期刊）
            scored_parts.sort(key=lambda x: x[1])
            # 跳过看起来像作者列表的候选，尝试下一个
            for part, score in scored_parts:
                cleaned = self._clean_title(part)
                if len(cleaned) < 15:
                    continue
                if self._is_likely_author_list(cleaned):
                    continue
                if len(cleaned.split()) <= 3 and len(cleaned) > 30:
                    # OCR 无空格，跳过
                    continue
                ref['title'] = cleaned
                break

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
            'transactions', 'letters', 'review', 'workshop',
            'entropy', 'vision research', 'scientific american',
            'communications in', 'evolutionary computation',
            'siam review', 'mind ', 'wiley', 'mit press',
            'oxford university', 'cambridge', 'interdiscip'
        ]
        text_lower = text.lower()
        # 短文本 + 关键词 = 很可能是期刊名
        if len(text) < 100 and any(kw in text_lower for kw in venue_keywords):
            return True
        # 以期刊特征格式开头: "Entropy 15(1)" / "Vision Research 49(10)"
        if re.match(r'^[A-Z][a-z]+ (?:Research|Review|Computation|Physics|Science)', text):
            return True
        return False

    def _clean_title(self, title: str) -> str:
        """清理标题"""
        # 移除开头的数字和标点
        title = re.sub(r'^[\d\.\[\]\-\s]+', '', title)
        # 修复OCR断字: "learn- ing" → "learning", "processingsystems" → 保留
        title = re.sub(r'(\w)-\s+([a-z])', r'\1\2', title)
        # 移除 LaTeX 特殊字符
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)
        title = re.sub(r'[\\{}]', '', title)
        # 修复常见OCR数字错误: "2o19" → "2019", "20io" → "2010"
        title = re.sub(r'(?<=[0-9])o(?=[0-9])', '0', title)
        title = re.sub(r'(?<=[0-9])i(?=[0-9])', '1', title)
        title = re.sub(r'(?<=[0-9])O(?=[0-9])', '0', title)
        title = re.sub(r'(?<=[0-9])I(?=[0-9])', '1', title)
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

    def find_matching_paper(self, ref_title: str, ref_author: str = '',
                            ref_arxiv_id: str = '', threshold: float = 0.90) -> Optional[str]:
        """
        在arxiv元数据中查找匹配的论文

        Args:
            ref_title: 参考文献标题
            ref_author: 参考文献作者（可选，用于辅助匹配）
            ref_arxiv_id: 参考文献中的arXiv ID（可选，最可靠的匹配方式）
            threshold: 相似度阈值

        Returns:
            Optional[str]: 匹配的arxiv_id，如果没找到则返回None
        """
        if not self.metadata_loaded:
            logger.warning("元数据未加载，无法进行匹配")
            return None

        # 0. arXiv ID直接匹配（最可靠的方式，参考zotero-reference的identifier匹配）
        if ref_arxiv_id:
            canonical = self.arxiv_id_to_canonical.get(ref_arxiv_id)
            if canonical:
                return canonical

        normalized_ref = self._normalize_title(ref_title)

        if len(normalized_ref) < 10:
            return None

        # 1. 精确匹配
        if normalized_ref in self.title_to_id:
            return self.title_to_id[normalized_ref]

        # 2. 部分匹配 - 使用倒排索引快速筛选候选
        ref_words = set(normalized_ref.split())
        stop_words = {'a', 'an', 'the', 'of', 'in', 'for', 'and', 'on', 'to', 'with',
                      'by', 'at', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                      'as', 'or', 'not', 'no', 'but', 'its', 'it', 'this', 'that',
                      'we', 'our', 'their', 'can', 'will', 'all', 'using', 'based'}
        significant_words = ref_words - stop_words
        significant_words = {w for w in significant_words if len(w) >= 3}

        if len(significant_words) < 2:
            return None

        # 通过倒排索引找到包含至少一个关键词的标题
        candidate_titles: Dict[str, int] = {}  # title -> overlap_count
        for word in significant_words:
            if word in self._word_index:
                for title in self._word_index[word]:
                    candidate_titles[title] = candidate_titles.get(title, 0) + 1

        # 只保留至少有2个词匹配的候选
        candidates = []
        for title, overlap in candidate_titles.items():
            if overlap >= 2:
                # 长度过滤
                if abs(len(title) - len(normalized_ref)) > len(normalized_ref) * 0.5:
                    continue
                arxiv_id = self.title_to_id[title]
                candidates.append((title, arxiv_id, overlap))

        # 按共同词数量排序，优先匹配相似度高的
        candidates.sort(key=lambda x: x[2], reverse=True)

        best_match = None
        best_score = 0

        # 只检查前500个候选
        for title, arxiv_id, _ in candidates[:500]:
            # 2a. 双向子串匹配（快速，参考zotero-reference的substring containment）
            if len(title) > 10 and len(normalized_ref) > 10:
                if normalized_ref in title or title in normalized_ref:
                    ratio = min(len(normalized_ref), len(title)) / max(len(normalized_ref), len(title))
                    if ratio > 0.75:
                        # 验证关键词重叠（防止短标题误匹配到不相关论文）
                        if self._validate_keyword_overlap(normalized_ref, title, 0.35):
                            return arxiv_id

            # 2b. SequenceMatcher模糊匹配（慢但精确）
            score = SequenceMatcher(None, normalized_ref, title).ratio()
            if score > best_score and score >= threshold:
                # 验证关键词重叠
                if self._validate_keyword_overlap(normalized_ref, title, 0.30):
                    best_score = score
                    best_match = arxiv_id
                    if score >= 0.95:
                        return best_match

        return best_match

    def _validate_keyword_overlap(self, normalized_ref: str, normalized_candidate: str,
                                   min_overlap: float = 0.30) -> bool:
        """
        验证两个标准化标题之间的关键词重叠度
        用于过滤子串匹配和模糊匹配中的误匹配
        """
        stop_words = {'a', 'an', 'the', 'of', 'in', 'for', 'and', 'on', 'to', 'with',
                      'by', 'at', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                      'as', 'or', 'not', 'no', 'but', 'its', 'it', 'this', 'that',
                      'we', 'our', 'their', 'can', 'will', 'all', 'using', 'based'}
        ref_words = set(normalized_ref.split()) - stop_words
        cand_words = set(normalized_candidate.split()) - stop_words
        ref_words = {w for w in ref_words if len(w) >= 3}
        cand_words = {w for w in cand_words if len(w) >= 3}
        if not ref_words or not cand_words:
            return True
        overlap = ref_words & cand_words
        min_size = min(len(ref_words), len(cand_words))
        return len(overlap) / max(min_size, 1) >= min_overlap

    # ==================== PDF直接提取引用 ====================

    def _detect_column_boundary(self, page) -> Optional[float]:
        """
        检测双列排版的列边界（参考zotero-reference的x坐标列检测算法）
        返回列间隙的中心x坐标，如果无法检测则返回None
        """
        chars = page.chars
        if not chars:
            return None

        width = page.width
        mid = width / 2

        # 在页面中间40%区域搜索间隙
        search_left = mid - width * 0.2
        search_right = mid + width * 0.2

        mid_chars = [c for c in chars
                     if search_left <= c['x0'] <= search_right and c['text'].strip()]
        if not mid_chars:
            return None

        mid_chars.sort(key=lambda c: c['x0'])

        # 找到最大的x间隔
        max_gap = 0
        gap_center = mid
        for i in range(len(mid_chars) - 1):
            gap = mid_chars[i + 1]['x0'] - mid_chars[i]['x1']
            if gap > max_gap:
                max_gap = gap
                gap_center = (mid_chars[i]['x1'] + mid_chars[i + 1]['x0']) / 2

        # 间隔足够大才认为是双列
        if max_gap > 12:
            return gap_center
        return None

    def _extract_page_text_with_columns(self, page) -> str:
        """
        从页面提取文本，自动检测并处理单列/双列布局
        参考zotero-reference的分栏算法：用x坐标间隙检测列边界
        """
        boundary = self._detect_column_boundary(page)

        if boundary:
            # 双列：分别裁剪左右列提取文本
            left_crop = page.crop((0, 0, boundary - 2, page.height))
            right_crop = page.crop((boundary + 2, 0, page.width, page.height))

            left_text = left_crop.extract_text() or ''
            right_text = right_crop.extract_text() or ''

            return left_text + '\n' + right_text
        else:
            return page.extract_text() or ''

    def _remove_headers_footers(self, page, all_pages_text: List[str], page_idx: int) -> str:
        """
        移除PDF页面中重复出现的页眉页脚
        参考zotero-reference的跨页位置匹配方法
        """
        text = page.extract_text() or ''
        if not text or page_idx >= len(all_pages_text):
            return text

        lines = text.split('\n')
        if len(lines) <= 2:
            return text

        # 检查首行和末行是否在其他页面重复出现
        first_line = lines[0].strip()
        last_line = lines[-1].strip()

        filtered = list(lines)

        # 首行重复检测（页眉）
        if first_line and len(first_line) < 80:
            repeat_count = 0
            for other_text in all_pages_text:
                if other_text and other_text.startswith(first_line):
                    repeat_count += 1
            if repeat_count >= 2:
                filtered = filtered[1:]

        # 末行重复检测（页脚/页码）
        if last_line and len(last_line) < 80:
            repeat_count = 0
            for other_text in all_pages_text:
                if other_text and other_text.rstrip().endswith(last_line):
                    repeat_count += 1
            if repeat_count >= 2:
                filtered = filtered[:-1]

        return '\n'.join(filtered)

    def extract_references_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        直接从PDF提取References部分文本（使用pdfplumber）
        改进：自动检测双列排版并分列提取，移除页眉页脚

        Args:
            pdf_path: PDF文件路径

        Returns:
            Optional[str]: References部分文本
        """
        if not HAS_PDFPLUMBER:
            return None

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # 预提取所有页面文本（用于页眉页脚检测）
                all_pages_text = []
                for page in pdf.pages:
                    all_pages_text.append(page.extract_text() or '')

                # 找到References/Bibliography起始页
                ref_start_page = None
                for i, page in enumerate(pdf.pages):
                    text = all_pages_text[i]
                    if re.search(r'\bReferences?\b|\bBibliography\b|\bREFERENCES\b', text, re.IGNORECASE):
                        ref_start_page = i
                        break

                if ref_start_page is None:
                    return None

                # 提取References起始页及之后页面的文本
                ref_texts = []
                for i in range(ref_start_page, total_pages):
                    page = pdf.pages[i]

                    # 使用列感知提取（自动检测双列布局）
                    text = self._extract_page_text_with_columns(page)

                    # 移除页眉页脚
                    text = self._remove_headers_footers(page, all_pages_text, i)

                    # 第一页：从References关键词开始截取
                    if i == ref_start_page:
                        match = re.search(
                            r'(?:References|Bibliography|REFERENCES)\s*\n',
                            text, re.IGNORECASE
                        )
                        if match:
                            text = text[match.end():]
                        else:
                            match = re.search(
                                r'(?:References|Bibliography|REFERENCES)',
                                text, re.IGNORECASE
                            )
                            if match:
                                text = text[match.end():]

                    # 遇到附录等章节停止
                    if i > ref_start_page and re.search(
                        r'\n(?:Appendix|A\s+|Acknowledgment|Supplementary)',
                        text, re.IGNORECASE
                    ):
                        appendix_match = re.search(
                            r'\n(?:Appendix|\d+\.\s*[A-Z]|Acknowledgment|Supplementary)',
                            text, re.IGNORECASE
                        )
                        if appendix_match:
                            text = text[:appendix_match.start()]

                    ref_texts.append(text)

                combined = '\n'.join(ref_texts).strip()
                return combined if len(combined) > 50 else None

        except Exception as e:
            logger.debug(f"pdfplumber提取References失败: {e}")
            return None

    def _is_likely_author_list(self, text: str) -> bool:
        """判断文本是否更像作者列表而不是论文标题"""
        text = text.strip()
        if not text:
            return False

        # "et al." → 几乎肯定是作者列表
        if re.search(r'\bet\s*al\.?', text, re.IGNORECASE):
            return True

        # 多个 "LastName, Initial." 模式
        name_initial = re.findall(r'[A-Z][a-z]+,\s*[A-Z]\.', text)
        if len(name_initial) >= 2:
            return True

        # "FirstName LastName and FirstName LastName" 模式
        # 但不是标题中的 "X and Y"（如 "Deep Learning and Neural Networks"）
        parts = re.split(r'\s+and\s+', text)
        if len(parts) >= 2:
            all_proper_nouns = True
            for part in parts:
                words = part.strip().split()
                if not words:
                    all_proper_nouns = False
                    break
                # 检查所有有意义词是否首字母大写（专有名词）
                significant = [w for w in words if len(w) > 2]
                if significant and not all(w[0].isupper() for w in significant):
                    all_proper_nouns = False
                    break
                # 平均词长 > 12 的不太可能是人名
                avg_len = sum(len(w) for w in words) / max(len(words), 1)
                if avg_len > 12:
                    all_proper_nouns = False
                    break
            if all_proper_nouns:
                return True

        # 高密度缩写：3个以上独立缩写 "A." "B." "C."
        initials = re.findall(r'\b[A-Z]\.\s', text)
        if len(initials) >= 3:
            return True

        return False

    def _is_garbled_title(self, title: str) -> bool:
        """判断标题是否为乱码/URL/期刊名/作者列表等无效内容"""
        # URL
        if re.search(r'https?://|doi\.org|doi\.0rg', title, re.IGNORECASE):
            return True
        # 纯DOI
        if re.match(r'^DOI:', title, re.IGNORECASE) or re.match(r'^D0I:', title):
            return True
        # 数学公式残留
        if re.search(r'\$[sv]\d|\\frac|\\displaystyle|\\begin\{|cid:', title):
            return True
        # 包含等号/数学符号（双列交错残留）
        if re.search(r'[→←∈∀∃∑∏∫≈≠≤≥Φ˜]', title):
            return True
        # 纯数字/符号/页码格式
        if re.match(r'^[\d\s,.\-$/\\]+$', title):
            return True
        if re.match(r'^pp\.\s*\d', title, re.IGNORECASE):
            return True
        # 过短
        if len(title) < 8:
            return True
        # 作者列表检测
        if self._is_likely_author_list(title):
            return True
        # OCR 无空格人名: "KinKwanLeungetal", "ScottBarnettetal", "AlexeyDosovitskiy"
        # 特征: 多个大写字母在单词内部 (CamelCase), 或含 "etal" 后缀
        if re.search(r'[a-z][A-Z][a-z]', title):
            # CamelCase 模式 - 检查是否是OCR合并的人名
            if re.search(r'etal$', title, re.IGNORECASE):
                return True
            # 全 CamelCase 无空格 (人名合并)
            if len(title.split()) <= 2 and len(title) > 15:
                cap_transitions = len(re.findall(r'[a-z][A-Z]', title))
                if cap_transitions >= 2:
                    return True
        # OCR 无空格: "completelyblind", "goingonavacation"
        # 特征: 一个很长的全小写单词
        if re.search(r'[a-z]{20,}', title):
            return True
        # OCR 无空格：长文本极少空格 → OCR合并或双列交错
        words = title.split()
        if len(words) <= 3 and len(title) > 30:
            return True
        # 短片段 + 包含逗号后跟数字/年份 → 不是标题
        # "Simple online in2021,", ",2 visionlearners"
        if re.search(r',\s*\d', title) and len(title) < 30:
            return True
        if re.match(r'^,\d', title):
            return True
        # 年份嵌入单词中: "in2021", "of2019" → OCR合并的碎片
        if re.search(r'[a-z]\d{4}', title) and len(title) < 30:
            return True
        # 单个人名: 1-3个词，全部首字母大写，像专有名词
        # "Mihailo Stojnic", "Alexey Dosovitskiy"
        twords = title.split()
        if 2 <= len(twords) <= 3 and all(w[0].isupper() for w in twords if w):
            if len(title) < 25:
                return True
        # 标题太短且不够有意义（少于3个有意义词）
        stop = {'a','an','the','of','in','for','and','on','to','with','by','at','from',
                'is','are','was','were','be','been','as','or','not','no','but','its'}
        sig_words = [w for w in re.split(r'\s+', title.lower()) if w and w not in stop and len(w) >= 3]
        if len(sig_words) < 3 and len(title) < 25:
            return True
        # 大量连续名字格式
        comma_count = title.count(',')
        if comma_count >= 4 and 'and ' in title:
            return True
        # "Proceedings of ..." 或 "In ..." 等纯期刊/会议名
        if re.match(r'^(Proceedings|Journal|Conference|International|Advances in '
                     r'Neural|Transactions|In\s)', title, re.IGNORECASE):
            if not re.search(r'\d{4}', title) and len(title) < 50:
                return True
        # "arXiv preprint arXiv:..." 无实际标题
        if re.match(r'^arXiv\s*(preprint\s*)?(arXiv:)?:?\s*\d', title, re.IGNORECASE):
            return True
        return False

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
        seen_edges: Set[Tuple[str, str]] = set()
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

            # 推断PDF路径
            pdf_path = paper_info.get('pdf_path', '')
            if not pdf_path and markdown_path:
                pdf_path = markdown_path.replace('.md', '.pdf')

            # 收集所有候选引用，从多个来源提取
            all_candidates: List[Dict] = []

            # 来源1: pdfplumber直接提取（双列论文效果更好）
            if pdf_path and os.path.exists(pdf_path):
                pdf_text = self.extract_references_from_pdf(pdf_path)
                if pdf_text:
                    pdf_refs = self.parse_references(pdf_text)
                    all_candidates.extend(pdf_refs)
                    logger.debug(f"[{paper_id}] pdfplumber提取: {len(pdf_refs)} 条")

            # 来源2: markdown提取（对pdfplumber提取不足的论文作为补充）
            if os.path.exists(markdown_path):
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                md_text = self.extract_references_section(content)
                if md_text:
                    md_refs = self.parse_references(md_text)
                    all_candidates.extend(md_refs)
                    logger.debug(f"[{paper_id}] markdown提取: {len(md_refs)} 条")

            if not all_candidates:
                logger.warning(f"未找到References部分: {paper_id}")
                continue

            stats['papers_with_references'] += 1

            # 去重：合并多个来源的引用，按标准化标题去重
            seen_titles = set()
            unique_refs = []
            for ref in all_candidates:
                title = ref.get('title', '')
                if not title:
                    continue
                normalized = self._normalize_title(title)
                if normalized and normalized not in seen_titles:
                    seen_titles.add(normalized)
                    unique_refs.append(ref)
            dup_count = len(all_candidates) - len(unique_refs)
            if dup_count > 0:
                logger.info(f"[{paper_id}] 合并去重: {len(all_candidates)} → {len(unique_refs)} 条引用 (去除 {dup_count} 条重复)")
            references = unique_refs
            stats['total_references_extracted'] += len(references)

            # 匹配并创建边
            for ref in references:
                ref_title = ref.get('title', '')

                # 提取arxiv_id
                ref_arxiv_id = ref.get('arxiv_id', '')

                # 乱码/无效标题检查
                title_is_garbled = self._is_garbled_title(ref_title) if ref_title else True
                if not ref_title or title_is_garbled:
                    # 即使标题无效，如果有arxiv_id（由_parse_single_ref_text提取）仍可匹配
                    if not ref_arxiv_id:
                        continue

                # 先尝试仅用标题匹配
                matched_id = None
                if ref_title and not title_is_garbled:
                    matched_id = self.find_matching_paper(ref_title, ref_arxiv_id='')

                # 标题未匹配且有arxiv_id时，用arxiv_id匹配
                # 交叉验证：arxiv_id指向的论文标题必须与引用文本有一定关键词重叠
                if not matched_id and ref_arxiv_id:
                    arxiv_match_id = self.find_matching_paper(
                        ref_title, ref_arxiv_id=ref_arxiv_id)
                    if arxiv_match_id:
                        # 用_raw_text做验证（比提取的标题更可靠）
                        raw_text = ref.get('_raw_text', ref_title)
                        matched_title = self.id_to_title.get(arxiv_match_id, '')
                        if matched_title and raw_text:
                            if not self._validate_keyword_overlap(
                                    self._normalize_title(raw_text),
                                    self._normalize_title(matched_title), 0.40):
                                continue
                        matched_id = arxiv_match_id
                if matched_id:
                    stats['matched_references'] += 1

                    # 去重：同一 source->target 边只保留一条
                    edge_key = (paper_id, matched_id)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

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
                - parallel: 是否并行处理
                - num_workers: 并行worker数
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
            parallel = task.get('parallel', False)
            num_workers = task.get('num_workers', 8)

            # 加载arxiv元数据
            if not self.metadata_loaded:
                self.load_arxiv_metadata(metadata_path)

            # 从上游数据获取论文列表
            papers = []
            if upstream_data:
                if 'papers' in upstream_data:
                    for paper in upstream_data['papers']:
                        papers.append({
                            'paper_id': paper.get('arxiv_id', paper.get('id', '')),
                            'title': paper.get('title', ''),
                            'markdown_path': paper.get('markdown_path', ''),
                            'pdf_path': paper.get('path', '')
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
            if parallel and len(papers) > 1:
                result = self._build_reference_graph_parallel(papers, output_csv, num_workers)
            else:
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

    def _build_reference_graph_parallel(
        self,
        paper_data: List[Dict[str, Any]],
        output_csv_path: str,
        num_workers: int = 8
    ) -> Dict[str, Any]:
        """并行构建引用关系图"""
        global _shared_title_to_id, _shared_id_to_title
        global _shared_arxiv_id_to_canonical, _shared_word_index

        # 设置模块级共享状态（fork子进程继承，COW零拷贝）
        _shared_title_to_id = self.title_to_id
        _shared_id_to_title = self.id_to_title
        _shared_arxiv_id_to_canonical = self.arxiv_id_to_canonical
        _shared_word_index = self._word_index

        logger.info(f"[并行引用图] 处理 {len(paper_data)} 篇论文, {num_workers} 个worker")

        with Pool(processes=num_workers) as pool:
            all_edge_lists = pool.map(_process_single_paper_refs, paper_data)

        # 合并和去重
        edges = []
        seen_edges: Set[Tuple[str, str]] = set()
        for edge_list in all_edge_lists:
            for edge in edge_list:
                edge_key = (edge['source_id'], edge['target_id'])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(edge)

        # 保存
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['source_id', 'target_id', 'source_title', 'target_title', 'reference_text'])
            writer.writeheader()
            writer.writerows(edges)

        stats = {
            'total_papers': len(paper_data),
            'papers_with_references': sum(1 for el in all_edge_lists if el),
            'total_references_extracted': sum(len(el) for el in all_edge_lists),
            'matched_references': sum(len(el) for el in all_edge_lists),
            'edges_created': len(edges)
        }

        logger.info(f"[并行引用图] 完成: {len(edges)} 条边, 保存到 {output_csv_path}")

        return {
            'status': 'success',
            'edges_file': output_csv_path,
            'edge_count': len(edges),
            'stats': stats
        }

"""
数据清洗Agent - 负责对收集的数据进行清洗整理，统一格式
纯规则驱动，不使用 LLM，保证 100% 可靠

功能：
1. 删除重复段落
2. 修复换行和截断问题
3. 规范化公式格式
4. 统一引用格式（增强版：处理连接在一起的引用）
5. 修复表格和图片位置
"""
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import os
import re
import hashlib
import numpy as np
from multiprocessing import Pool

from .base_agent import BaseAgent


def _clean_paper_standalone(paper_dir: str) -> Optional[Dict]:
    """独立清洗函数，供multiprocessing.Pool调用（picklable）"""
    agent = DataCleanerAgent(client=None, system_prompt="")
    return agent._clean_paper_directory(paper_dir)


class DataCleanerAgent(BaseAgent):
    """数据清洗Agent - 纯规则驱动，不使用 LLM"""

    def __init__(
        self,
        client=None,
        system_prompt: str = "",
        mcp_tool_manager: Optional[Any] = None,
        model: str = None
    ):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="DataCleanerAgent"
        )
        self.cleaned_data: List[Dict] = []
        self.cleaning_stats: Dict = {}

    def execute(
        self,
        task: Dict[str, Any],
        upstream_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """执行数据清洗任务"""
        logger.info(f"[{self.agent_name}] 开始执行数据清洗任务")

        paper_dirs = self._get_paper_dirs(task, upstream_data)

        results = {
            "status": "success",
            "cleaned_papers": [],
            "stats": {
                "total_input": len(paper_dirs),
                "total_cleaned": 0,
                "errors": 0,
                "duplicates_removed": 0,
                "chars_fixed": 0
            },
            "errors": []
        }

        parallel = task.get("parallel", False)
        num_workers = task.get("num_workers", 8)

        if parallel and len(paper_dirs) > 1:
            logger.info(f"[{self.agent_name}] 并行清洗 {len(paper_dirs)} 篇, {num_workers} 个worker")
            with Pool(processes=num_workers) as pool:
                raw_results = pool.map(_clean_paper_standalone, paper_dirs)
            for result in raw_results:
                if result:
                    results["cleaned_papers"].append(result)
                    results["stats"]["total_cleaned"] += 1
                    results["stats"]["duplicates_removed"] += result.get("duplicates_removed", 0)
                    results["stats"]["chars_fixed"] += result.get("chars_fixed", 0)
                else:
                    results["stats"]["errors"] += 1
        else:
            for paper_dir in paper_dirs:
                try:
                    result = self._clean_paper_directory(paper_dir)
                    if result:
                        results["cleaned_papers"].append(result)
                        results["stats"]["total_cleaned"] += 1
                        results["stats"]["duplicates_removed"] += result.get("duplicates_removed", 0)
                        results["stats"]["chars_fixed"] += result.get("chars_fixed", 0)
                except Exception as e:
                    logger.error(f"清洗论文目录失败 {paper_dir}: {e}")
                    results["errors"].append(f"{paper_dir}: {str(e)}")
                    results["stats"]["errors"] += 1

        self.cleaned_data = results["cleaned_papers"]
        self.cleaning_stats = results["stats"]

        logger.info(
            f"[{self.agent_name}] 数据清洗完成: "
            f"输入 {results['stats']['total_input']}, "
            f"成功 {results['stats']['total_cleaned']}"
        )

        return results

    def _get_paper_dirs(self, task: Dict, upstream_data: Optional[Dict]) -> List[str]:
        """获取论文目录列表"""
        paper_dirs = []

        if task.get("paper_dirs"):
            paper_dirs = task["paper_dirs"]
        elif upstream_data:
            papers = upstream_data.get("papers", [])
            for paper in papers:
                if "paper_dir" in paper:
                    paper_dirs.append(paper["paper_dir"])
                elif "markdown_path" in paper:
                    paper_dirs.append(os.path.dirname(paper["markdown_path"]))
        else:
            default_dir = task.get("data_dir", "./data/recent_papers")
            if os.path.exists(default_dir):
                for item in os.listdir(default_dir):
                    item_path = os.path.join(default_dir, item)
                    if os.path.isdir(item_path):
                        paper_dirs.append(item_path)

        return paper_dirs

    def _clean_paper_directory(self, paper_dir: str) -> Optional[Dict]:
        """清洗单个论文目录"""
        paper_id = os.path.basename(paper_dir)
        logger.info(f"清洗论文: {paper_id}")

        md_path = os.path.join(paper_dir, f"{paper_id}.md")
        npy_path = os.path.join(paper_dir, f"{paper_id}.npy")

        if not os.path.exists(md_path):
            logger.warning(f"Markdown 文件不存在: {md_path}")
            return None

        with open(md_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        original_length = len(raw_content)

        # 执行清洗管道
        cleaned_content, stats = self._clean_pipeline(raw_content)

        # 保存清洗后的内容
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        # 更新 npy 文件
        if os.path.exists(npy_path):
            self._update_npy_file(npy_path, cleaned_content)

        logger.info(f"论文 {paper_id} 清洗完成")

        return {
            "status": "success",
            "paper_id": paper_id,
            "paper_dir": paper_dir,
            "markdown_path": md_path,
            "original_length": original_length,
            "cleaned_length": len(cleaned_content),
            **stats
        }

    def _clean_pipeline(self, text: str) -> Tuple[str, Dict]:
        """清洗管道"""
        stats = {
            "duplicates_removed": 0,
            "chars_fixed": 0,
            "formulas_fixed": 0,
            "refs_formatted": 0
        }

        # 1. 修复连字符分割 (Hel- lo -> Hello)
        text, fix_count = self._fix_hyphenation(text)
        stats["chars_fixed"] += fix_count

        # 2. 修复换行问题
        text, fix_count = self._fix_line_breaks(text)
        stats["chars_fixed"] += fix_count

        # 3. 删除重复段落
        text, dup_count = self._remove_duplicate_paragraphs(text)
        stats["duplicates_removed"] = dup_count

        # 4. 规范化公式格式
        text, fix_count = self._normalize_formulas(text)
        stats["formulas_fixed"] = fix_count

        # 5. 规范化引用格式（增强版）
        text, ref_count = self._normalize_references_enhanced(text)
        stats["refs_formatted"] = ref_count

        # 6. 清理多余空白
        text = self._cleanup_whitespace(text)

        return text, stats

    def _fix_hyphenation(self, text: str) -> Tuple[str, int]:
        """修复连字符分割"""
        fix_count = 0

        def replace_hyphen(match):
            nonlocal fix_count
            fix_count += 1
            return match.group(1) + match.group(2)

        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', replace_hyphen, text)
        # 也修复同一行内的连字符问题
        text = re.sub(r'(\w+)-\s+(\w+)', replace_hyphen, text)

        return text, fix_count

    def _fix_line_breaks(self, text: str) -> Tuple[str, int]:
        """修复 PDF 转换导致的换行问题"""
        fix_count = 0
        lines = text.split('\n')
        result = []
        i = 0

        while i < len(lines):
            line = lines[i]

            while i + 1 < len(lines):
                next_line = lines[i + 1]

                # 不合并的情况
                if not next_line.strip():
                    break
                if next_line.strip().startswith('#'):
                    break
                if next_line.strip().startswith(('- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ')):
                    break
                if re.match(r'^\[\d+\]', next_line.strip()):
                    break
                if re.match(r'^Figure \d+|^Table \d+', next_line.strip(), re.IGNORECASE):
                    break
                if line.strip().endswith(('.', '!', '?', ':', ';')):
                    break
                if re.match(r'^[A-Z][a-z]+,?\s+[A-Z]\.', next_line.strip()):
                    # 可能是引用的开始 (如 "Smith, J.")
                    break

                # 合并行
                if not line.endswith(' '):
                    line = line + ' ' + next_line
                else:
                    line = line + next_line
                fix_count += 1
                i += 1

            result.append(line)
            i += 1

        return '\n'.join(result), fix_count

    def _remove_duplicate_paragraphs(self, text: str) -> Tuple[str, int]:
        """删除重复的段落"""
        paragraphs = re.split(r'\n\n+', text)

        seen_hashes = set()
        unique_paragraphs = []
        dup_count = 0

        for para in paragraphs:
            normalized = ' '.join(para.split())
            if len(normalized) < 50:
                unique_paragraphs.append(para)
                continue

            para_hash = hashlib.md5(normalized.encode()).hexdigest()

            if para_hash not in seen_hashes:
                seen_hashes.add(para_hash)
                unique_paragraphs.append(para)
            else:
                dup_count += 1

        return '\n\n'.join(unique_paragraphs), dup_count

    def _normalize_formulas(self, text: str) -> Tuple[str, int]:
        """规范化 LaTeX 公式格式"""
        fix_count = 0
        original = text

        text = re.sub(r'\$\$\s+', '$$', text)
        text = re.sub(r'\s+\$\$', '$$', text)
        text = re.sub(r'(?<!\$)\$\s+([^\$]+?)\s+\$(?!\$)', r'$\1$', text)
        text = re.sub(r'([^\n])\$\$', r'\1\n$$', text)
        text = re.sub(r'\$\$([^\n])', r'$$\n\1', text)

        fix_count = abs(len(original) - len(text))
        return text, fix_count

    def _normalize_references_enhanced(self, text: str) -> Tuple[str, int]:
        """
        增强版引用格式规范化
        处理连接在一起的引用、多个References部分等问题
        """
        ref_count = 0

        # 找到所有可能的 References 位置
        ref_positions = []

        # 模式1: ## References
        for match in re.finditer(r'##\s*References?\s*\n', text, re.IGNORECASE):
            ref_positions.append(match.start())

        # 模式2: 单独的 References 行（后面紧跟引用内容）
        for match in re.finditer(r'\nReferences?\s+\n', text):
            ref_positions.append(match.start())

        # 模式3: References 后面直接跟作者名
        for match in re.finditer(r'\nReferences?\s+[A-Z][a-z]+', text):
            ref_positions.append(match.start())

        if not ref_positions:
            return text, ref_count

        # 取最后一个 References 位置作为真正的引用部分
        ref_start = max(ref_positions)

        # 找到 References 部分的结束（下一个 ## 标题或文件结束）
        after_ref = text[ref_start:]
        next_section = re.search(r'\n##\s+', after_ref)
        if next_section:
            ref_end = ref_start + next_section.start()
            ref_content = text[ref_start:ref_end]
            after_ref_text = text[ref_end:]
        else:
            ref_content = after_ref
            after_ref_text = ""

        # 提取并格式化引用
        formatted_refs = self._extract_and_format_references(ref_content)
        ref_count = len(formatted_refs)

        # 删除之前的所有 References 部分
        # 保留第一个 ## References 之前的内容
        first_ref_match = re.search(r'##\s*References?\s*\n', text, re.IGNORECASE)
        if first_ref_match:
            main_content = text[:first_ref_match.start()]
        else:
            # 找最后一个 ## 标题作为正文结束
            last_section = None
            for match in re.finditer(r'\n##\s+', text):
                if match.start() < ref_start:
                    last_section = match
            if last_section:
                main_content = text[:last_section.start()]
            else:
                main_content = text[:ref_start]

        # 重新组合
        if formatted_refs:
            new_ref_section = "\n## References\n\n" + '\n'.join(formatted_refs)
        else:
            new_ref_section = ""

        text = main_content + new_ref_section + after_ref_text

        return text, ref_count

    def _extract_and_format_references(self, content: str) -> List[str]:
        """
        从混乱的 References 内容中提取并格式化单条引用
        """
        refs = []

        # 移除 References 标题
        content = re.sub(r'^##\s*References?\s*\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^References?\s*\n', '', content, flags=re.IGNORECASE)

        # 清理内容
        content = ' '.join(content.split())

        # 方法1: 按作者名模式分割
        # 匹配 "LastName, FirstName" 或 "LastName, F." 模式
        author_pattern = r'(?=[A-Z][a-z]+,?\s+(?:[A-Z]\.|[A-Z][a-z]+))'

        # 先尝试按已有的 [数字] 标签分割
        if re.search(r'\[\d+\]', content):
            parts = re.split(r'(?=\[\d+\])', content)
            for part in parts:
                part = part.strip()
                if part and len(part) > 20:
                    # 移除标签
                    clean_ref = re.sub(r'^\[\d+\]\s*', '', part)
                    if self._is_valid_reference(clean_ref):
                        refs.append(clean_ref)

        # 如果按标签分割效果不好，按作者模式分割
        if len(refs) < 5:
            refs = []
            # 按作者名分割
            parts = re.split(author_pattern, content)

            current_ref = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # 检查是否是新引用的开始
                if re.match(r'^[A-Z][a-z]+,?\s+[A-Z]', part):
                    if current_ref and self._is_valid_reference(current_ref):
                        refs.append(current_ref)
                    current_ref = part
                else:
                    if current_ref:
                        current_ref += " " + part
                    else:
                        current_ref = part

            if current_ref and self._is_valid_reference(current_ref):
                refs.append(current_ref)

        # 方法2: 如果上面效果不好，尝试按年份分割
        if len(refs) < 5:
            refs = []
            # 按年份+句号分割
            year_pattern = r'(?<=\d{4})\.\s+(?=[A-Z][a-z]+,?\s+[A-Z])'
            parts = re.split(year_pattern, content)

            for part in parts:
                part = part.strip()
                if part and self._is_valid_reference(part):
                    refs.append(part)

        # 去重
        unique_refs = []
        seen = set()
        for ref in refs:
            key = ref.lower()[:80]
            if key not in seen and len(ref) > 30:
                seen.add(key)
                unique_refs.append(ref)

        # 格式化引用
        formatted_refs = []
        for i, ref in enumerate(unique_refs, 1):
            formatted = self._format_single_reference(ref, i)
            formatted_refs.append(formatted)

        return formatted_refs

    def _is_valid_reference(self, text: str) -> bool:
        """判断文本是否是一条有效的引用"""
        # 必须包含年份
        if not re.search(r'\d{4}', text):
            return False

        # 必须足够长
        if len(text) < 30:
            return False

        # 应该包含作者名模式或会议/期刊名
        has_author = re.search(r'[A-Z][a-z]+,?\s+[A-Z]', text)
        has_venue = re.search(r'(Conference|Journal|Proceedings|arXiv|IEEE|ACM|ICML|NeurIPS|CVPR|ICCV)', text, re.IGNORECASE)

        return bool(has_author or has_venue)

    def _format_single_reference(self, ref: str, index: int) -> str:
        """格式化单条引用"""
        # 清理
        ref = ' '.join(ref.split())

        # 提取 arXiv ID
        arxiv_match = re.search(r'arXiv[:\s]+(\d{4}\.\d{4,5})', ref, re.IGNORECASE)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        # 移除已有标签
        ref = re.sub(r'^\[\d+\]\s*', '', ref)

        # 清理引用内容
        ref = ref.strip()
        if not ref.endswith('.'):
            ref += '.'

        if arxiv_id:
            ref = re.sub(r'[,.]?\s*arXiv[:\s]+\d{4}\.\d{4,5}', '', ref, flags=re.IGNORECASE)
            ref = ref.rstrip('. ')
            return f"[{index}] {ref}. arXiv:{arxiv_id}."
        else:
            return f"[{index}] {ref}"

    def _cleanup_whitespace(self, text: str) -> str:
        """清理多余空白"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = text.rstrip() + '\n'
        return text

    def _update_npy_file(self, npy_path: str, cleaned_content: str):
        """更新 npy 文件"""
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            if isinstance(data, dict) and 'text' in data:
                data['text'] = cleaned_content
                np.save(npy_path, data)
                logger.info(f"已更新 NPY 文件: {npy_path}")
        except Exception as e:
            logger.warning(f"处理 NPY 文件失败 {npy_path}: {e}")

    def clean_single_paper(self, paper_dir: str) -> Dict:
        """清洗单个论文"""
        return self._clean_paper_directory(paper_dir)

    def get_cleaned_data(self) -> List[Dict]:
        """获取清洗后的数据"""
        return self.cleaned_data

    def get_cleaning_stats(self) -> Dict:
        """获取清洗统计"""
        return self.cleaning_stats

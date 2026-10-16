"""
数据清洗Agent - ReAct版
LLM驱动质量分析 + 规则管道执行

ReAct能力: 发现清洗问题→通知上游Collector重提取
清洗逻辑（_clean_pipeline等）完全不变
"""
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import os
import re
import hashlib
import numpy as np
from multiprocessing import Pool

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


def _clean_paper_standalone(paper_dir: str) -> Optional[Dict]:
    """独立清洗函数，供multiprocessing.Pool调用（picklable）"""
    agent = DataCleanerAgent(client=None, system_prompt="")
    return agent._clean_paper_directory(paper_dir)


class DataCleanerAgent(ReActAgent):
    """数据清洗Agent — ReAct循环驱动"""

    def __init__(self, client=None, system_prompt: str = "",
                 mcp_tool_manager=None, blackboard=None, message_bus=None,
                 model: str = None):
        super().__init__(
            client=client,
            system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager,
            agent_name="DataCleanerAgent",
            blackboard=blackboard,
            message_bus=message_bus
        )
        self.cleaned_data: List[Dict] = []
        self.cleaning_stats: Dict = {}

    # ==================== ReAct 接口实现 ====================

    def get_available_actions(self) -> List[str]:
        return [
            "discover_dirs",       # 发现需要清洗的论文目录
            "clean_paper",         # 清洗单篇论文
            "clean_batch",         # 批量清洗（并行）
            "validate_cleaned",    # 验证清洗结果
            "report_issues",       # 向上游报告质量问题
            "done"
        ]

    def is_done(self, observation: Dict) -> bool:
        results = observation.get("accumulated_results", {})
        total = results.get("stats", {}).get("total_input", 0)
        cleaned = results.get("stats", {}).get("total_cleaned", 0)
        errors = results.get("stats", {}).get("errors", 0)
        return total > 0 and (cleaned + errors >= total)

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        # 从blackboard读取collector结果
        if blackboard:
            extracted = blackboard.read("collector", "extracted_folders", [])
            observation["extracted_folders_count"] = len(extracted)
            observation["extracted_folders"] = extracted

        # 当前清洗进度
        results = observation.get("accumulated_results", {})
        stats = results.get("stats", {})
        observation["total_input"] = stats.get("total_input", 0)
        observation["total_cleaned"] = stats.get("total_cleaned", 0)
        observation["pending_count"] = stats.get("total_input", 0) - stats.get("total_cleaned", 0) - stats.get("errors", 0)

        return observation

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        if action == "discover_dirs":
            return self.action_discover_dirs(self._current_task, self._upstream_data)
        elif action == "clean_paper":
            return self.action_clean_paper(params.get("paper_dir", ""))
        elif action == "clean_batch":
            return self.action_clean_batch(self._current_task)
        elif action == "validate_cleaned":
            return self.action_validate_cleaned()
        elif action == "report_issues":
            return self.action_report_issues()
        elif action == "done":
            return {"status": "success", "message": "清洗完成"}

        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动方法 ====================

    def action_discover_dirs(self, task: Dict, upstream_data: Optional[Dict] = None) -> Dict:
        """发现需要清洗的论文目录"""
        paper_dirs = self._get_paper_dirs(task, upstream_data)

        if self.blackboard:
            self.blackboard.write("cleaner", "paper_dirs", paper_dirs)
            self.blackboard.write("cleaner", "total_input", len(paper_dirs))

        return {
            "status": "success",
            "paper_dirs": paper_dirs,
            "stats": {"total_input": len(paper_dirs)}
        }

    def action_clean_paper(self, paper_dir: str) -> Dict:
        """清洗单篇论文"""
        if not paper_dir or not os.path.exists(paper_dir):
            return {"status": "error", "message": f"目录不存在: {paper_dir}"}

        result = self._clean_paper_directory(paper_dir)
        if result:
            if self.blackboard:
                self.blackboard.write("cleaner", f"cleaned_{result.get('paper_id', '')}", result)

            # 更新累积统计
            stats = self._accumulated_results.get("stats", {
                "total_input": 0, "total_cleaned": 0, "errors": 0,
                "duplicates_removed": 0, "chars_fixed": 0
            })
            stats["total_cleaned"] = stats.get("total_cleaned", 0) + 1
            stats["duplicates_removed"] = stats.get("duplicates_removed", 0) + result.get("duplicates_removed", 0)
            stats["chars_fixed"] = stats.get("chars_fixed", 0) + result.get("chars_fixed", 0)

            return {
                "status": "success",
                "cleaned_paper": result,
                "stats": stats
            }
        return {"status": "error", "message": f"清洗失败: {paper_dir}"}

    def action_clean_batch(self, task: Dict) -> Dict:
        """批量清洗（支持并行）"""
        paper_dirs = self._accumulated_results.get("paper_dirs", [])
        if not paper_dirs:
            # 先发现目录
            dirs_result = self.action_discover_dirs(task, self._upstream_data)
            paper_dirs = dirs_result.get("paper_dirs", [])

        if not paper_dirs:
            return {"status": "error", "message": "没有找到需要清洗的目录"}

        parallel = task.get("parallel", False)
        num_workers = task.get("num_workers", 8)

        cleaned_papers = []
        all_errors = []
        total_dup = 0
        total_chars = 0

        if parallel and len(paper_dirs) > 1:
            logger.info(f"[{self.agent_name}] 并行清洗 {len(paper_dirs)} 篇, {num_workers} workers")
            with Pool(processes=num_workers) as pool:
                raw_results = pool.map(_clean_paper_standalone, paper_dirs)
            for result in raw_results:
                if result:
                    cleaned_papers.append(result)
                    total_dup += result.get("duplicates_removed", 0)
                    total_chars += result.get("chars_fixed", 0)
                else:
                    all_errors.append("清洗返回None")
        else:
            for paper_dir in paper_dirs:
                try:
                    result = self._clean_paper_directory(paper_dir)
                    if result:
                        cleaned_papers.append(result)
                        total_dup += result.get("duplicates_removed", 0)
                        total_chars += result.get("chars_fixed", 0)
                    else:
                        all_errors.append(f"{paper_dir}: 清洗返回None")
                except Exception as e:
                    all_errors.append(f"{paper_dir}: {str(e)}")

        self.cleaned_data = cleaned_papers
        stats = {
            "total_input": len(paper_dirs),
            "total_cleaned": len(cleaned_papers),
            "errors": len(all_errors),
            "duplicates_removed": total_dup,
            "chars_fixed": total_chars
        }
        self.cleaning_stats = stats

        if self.blackboard:
            self.blackboard.write("cleaner", "stats", stats)
            self.blackboard.write("cleaner", "cleaned_papers", cleaned_papers)

        return {
            "status": "success",
            "cleaned_papers": cleaned_papers,
            "errors": all_errors,
            "stats": stats
        }

    def action_validate_cleaned(self) -> Dict:
        """验证清洗结果"""
        issues = []
        cleaned = self._accumulated_results.get("cleaned_papers", [])

        for paper in cleaned:
            paper_dir = paper.get("paper_dir", "")
            md_path = os.path.join(paper_dir, f"{paper.get('paper_id', '')}.md")

            if not os.path.exists(md_path):
                issues.append({"paper_dir": paper_dir, "issue": "清洗后md文件不存在"})
                continue

            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_len = paper.get("original_length", 0)
            cleaned_len = len(content)

            # 检查异常缩小
            if original_len > 1000 and cleaned_len < original_len * 0.2:
                issues.append({
                    "paper_dir": paper_dir,
                    "issue": f"文件异常缩小: {original_len} -> {cleaned_len} ({cleaned_len/original_len*100:.0f}%)",
                    "severity": "warning"
                })

        if issues and self.message_bus:
            self.message_bus.send(Message(
                sender=self.agent_name,
                recipient="PaperCollectorAgent",
                msg_type="quality_issue",
                content={"cleaning_issues": issues},
                priority=1
            ))

        return {
            "status": "success",
            "issues_count": len(issues),
            "issues": issues
        }

    def action_report_issues(self) -> Dict:
        """向上游Agent报告质量问题"""
        issues = self._accumulated_results.get("issues", [])
        if not issues:
            return {"status": "success", "message": "没有需要报告的问题"}

        if self.message_bus:
            self.message_bus.send(Message(
                sender=self.agent_name,
                recipient="PaperCollectorAgent",
                msg_type="quality_issue",
                content={"issues": issues},
                priority=1
            ))

        return {"status": "success", "message": f"已报告 {len(issues)} 个问题"}

    # ==================== execute 入口（兼容旧接口）====================

    def execute(self, task: Dict[str, Any],
                upstream_data: Optional[Dict] = None) -> Dict[str, Any]:
        self._current_task = task
        self._upstream_data = upstream_data

        if self.blackboard and self.message_bus:
            result = self.run_react_loop(task, self.blackboard, self.message_bus)
            self._ensure_compatible_result(result)
            return result

        return self._execute_legacy(task, upstream_data)

    def _ensure_compatible_result(self, result: Dict):
        """确保ReAct结果与旧格式兼容"""
        if "cleaned_papers" not in result:
            result["cleaned_papers"] = self._accumulated_results.get("cleaned_papers", [])
        if "stats" not in result:
            result["stats"] = self._accumulated_results.get("stats", {
                "total_input": 0, "total_cleaned": 0, "errors": 0,
                "duplicates_removed": 0, "chars_fixed": 0
            })
        if "errors" not in result:
            result["errors"] = self._accumulated_results.get("errors", [])

        evaluation = self.evaluate_own_results(result)

    def _execute_legacy(self, task: Dict, upstream_data=None) -> Dict:
        """原始执行逻辑（向后兼容）"""
        logger.info(f"[{self.agent_name}] 开始执行数据清洗任务")

        analysis = self.reason_about_task(task)
        logger.info(f"[{self.agent_name}] 任务理解: {analysis.get('understanding', '')}")

        paper_dirs = self._get_paper_dirs(task, upstream_data)

        results = {
            "status": "success",
            "cleaned_papers": [],
            "stats": {
                "total_input": len(paper_dirs),
                "total_cleaned": 0, "errors": 0,
                "duplicates_removed": 0, "chars_fixed": 0
            },
            "errors": []
        }

        parallel = task.get("parallel", False)
        num_workers = task.get("num_workers", 8)

        if parallel and len(paper_dirs) > 1:
            logger.info(f"[{self.agent_name}] 并行清洗 {len(paper_dirs)} 篇, {num_workers} workers")
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

        evaluation = self.evaluate_own_results(results)
        logger.info(
            f"[{self.agent_name}] 数据清洗完成: 输入 {results['stats']['total_input']}, "
            f"成功 {results['stats']['total_cleaned']}"
        )

        return results

    # ==================== 核心清洗逻辑（完全不变）====================

    def _get_paper_dirs(self, task: Dict, upstream_data: Optional[Dict]) -> List[str]:
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
                elif "path" in paper:
                    paper_dirs.append(os.path.dirname(paper["path"]))
        else:
            default_dir = task.get("data_dir", "./data/papers")
            if os.path.exists(default_dir):
                for item in os.listdir(default_dir):
                    item_path = os.path.join(default_dir, item)
                    if os.path.isdir(item_path):
                        paper_dirs.append(item_path)
        return paper_dirs

    def _clean_paper_directory(self, paper_dir: str) -> Optional[Dict]:
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
        cleaned_content, stats = self._clean_pipeline(raw_content)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

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
        stats = {
            "duplicates_removed": 0, "chars_fixed": 0,
            "formulas_fixed": 0, "refs_formatted": 0
        }
        text, fix_count = self._fix_hyphenation(text)
        stats["chars_fixed"] += fix_count
        text, fix_count = self._fix_line_breaks(text)
        stats["chars_fixed"] += fix_count
        text, dup_count = self._remove_duplicate_paragraphs(text)
        stats["duplicates_removed"] = dup_count
        text, fix_count = self._normalize_formulas(text)
        stats["formulas_fixed"] += fix_count
        text, ref_count = self._normalize_references_enhanced(text)
        stats["refs_formatted"] = ref_count
        text = self._cleanup_whitespace(text)
        return text, stats

    def _fix_hyphenation(self, text: str) -> Tuple[str, int]:
        fix_count = 0
        def replace_hyphen(match):
            nonlocal fix_count
            fix_count += 1
            return match.group(1) + match.group(2)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', replace_hyphen, text)
        text = re.sub(r'(\w+)-\s+(\w+)', replace_hyphen, text)
        return text, fix_count

    def _fix_line_breaks(self, text: str) -> Tuple[str, int]:
        fix_count = 0
        lines = text.split('\n')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            while i + 1 < len(lines):
                next_line = lines[i + 1]
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
                    break
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
        ref_count = 0
        ref_positions = []
        for match in re.finditer(r'##\s*References?\s*\n', text, re.IGNORECASE):
            ref_positions.append(match.start())
        for match in re.finditer(r'\nReferences?\s+\n', text):
            ref_positions.append(match.start())
        for match in re.finditer(r'\nReferences?\s+[A-Z][a-z]+', text):
            ref_positions.append(match.start())
        if not ref_positions:
            return text, ref_count
        ref_start = max(ref_positions)
        after_ref = text[ref_start:]
        next_section = re.search(r'\n##\s+', after_ref)
        if next_section:
            ref_end = ref_start + next_section.start()
            ref_content = text[ref_start:ref_end]
            after_ref_text = text[ref_end:]
        else:
            ref_content = after_ref
            after_ref_text = ""
        formatted_refs = self._extract_and_format_references(ref_content)
        ref_count = len(formatted_refs)
        first_ref_match = re.search(r'##\s*References?\s*\n', text, re.IGNORECASE)
        if first_ref_match:
            main_content = text[:first_ref_match.start()]
        else:
            last_section = None
            for match in re.finditer(r'\n##\s+', text):
                if match.start() < ref_start:
                    last_section = match
            if last_section:
                main_content = text[:last_section.start()]
            else:
                main_content = text[:ref_start]
        if formatted_refs:
            new_ref_section = "\n## References\n\n" + '\n'.join(formatted_refs)
        else:
            new_ref_section = ""
        text = main_content + new_ref_section + after_ref_text
        return text, ref_count

    def _extract_and_format_references(self, content: str) -> List[str]:
        refs = []
        content = re.sub(r'^##\s*References?\s*\n', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^References?\s*\n', '', content, flags=re.IGNORECASE)
        content = ' '.join(content.split())
        author_pattern = r'(?=[A-Z][a-z]+,?\s+(?:[A-Z]\.|[A-Z][a-z]+))'
        if re.search(r'\[\d+\]', content):
            parts = re.split(r'(?=\[\d+\])', content)
            for part in parts:
                part = part.strip()
                if part and len(part) > 20:
                    clean_ref = re.sub(r'^\[\d+\]\s*', '', part)
                    if self._is_valid_reference(clean_ref):
                        refs.append(clean_ref)
        if len(refs) < 5:
            refs = []
            parts = re.split(author_pattern, content)
            current_ref = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
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
        if len(refs) < 5:
            refs = []
            year_pattern = r'(?<=\d{4})\.\s+(?=[A-Z][a-z]+,?\s+[A-Z])'
            parts = re.split(year_pattern, content)
            for part in parts:
                part = part.strip()
                if part and self._is_valid_reference(part):
                    refs.append(part)
        unique_refs = []
        seen = set()
        for ref in refs:
            key = ref.lower()[:80]
            if key not in seen and len(ref) > 30:
                seen.add(key)
                unique_refs.append(ref)
        formatted_refs = []
        for i, ref in enumerate(unique_refs, 1):
            formatted = self._format_single_reference(ref, i)
            formatted_refs.append(formatted)
        return formatted_refs

    def _is_valid_reference(self, text: str) -> bool:
        if not re.search(r'\d{4}', text):
            return False
        if len(text) < 30:
            return False
        has_author = re.search(r'[A-Z][a-z]+,?\s+[A-Z]', text)
        has_venue = re.search(
            r'(Conference|Journal|Proceedings|arXiv|IEEE|ACM|ICML|NeurIPS|CVPR|ICCV)',
            text, re.IGNORECASE
        )
        return bool(has_author or has_venue)

    def _format_single_reference(self, ref: str, index: int) -> str:
        ref = ' '.join(ref.split())
        arxiv_match = re.search(r'arXiv[:\s]+(\d{4}\.\d{4,5})', ref, re.IGNORECASE)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        ref = re.sub(r'^\[\d+\]\s*', '', ref)
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
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = text.rstrip() + '\n'
        return text

    def _update_npy_file(self, npy_path: str, cleaned_content: str):
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            if isinstance(data, dict) and 'text' in data:
                data['text'] = cleaned_content
                np.save(npy_path, data)
                logger.info(f"已更新 NPY 文件: {npy_path}")
        except Exception as e:
            logger.warning(f"处理 NPY 文件失败 {npy_path}: {e}")

    def clean_single_paper(self, paper_dir: str) -> Dict:
        return self._clean_paper_directory(paper_dir)

    def get_cleaned_data(self) -> List[Dict]:
        return self.cleaned_data

    def get_cleaning_stats(self) -> Dict:
        return self.cleaning_stats

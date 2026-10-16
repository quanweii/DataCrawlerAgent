"""
agents.domain_collector — 参数化 Collector Agent（通用跨域采集基类）。

仿 DomainGraphBuilderAgent 设计：所有 ReAct 循环逻辑共享，域差异通过 config dict
表达。对 file-based 数据源（已有 JSON）无需任何自定义 Python 代码。

Config dict 键:
    domain:               str   — 域名称 (e.g. "steam")
    agent_name:           str   — agent 标识 (e.g. "SteamGameCollectorAgent")
    id_field:             str   — item 唯一 ID 字段 (e.g. "appid")
    label_field:          str   — 标签字段 (e.g. "genres[0]", "style")
    data_source:          str   — "file" (默认) | "api"
    data_file:            str   — JSON 数据文件路径
    data_dir:             str   — 数据目录
    max_items:            int   — 最大加载 item 数
    collector_ns:         str   — blackboard 写命名空间 (e.g. "steam_collector")
    cslo_dir_key:         str   — output_dirs 中 cs_lo 路径的键 (e.g. "items")
    tool_manager_class:   str   — 可选: fully qualified 工具管理器类路径
    max_items_param:      str   — 可选: 传给 tool_manager.load_raw 的参数名 (e.g. "max_games")
"""

import os
import json
import importlib
from typing import Dict, Any, List, Optional
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class DomainCollectorAgent(ReActAgent):
    """通用参数化 Collector Agent — 对 file-based 域和 API-based 域均可工作。"""

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="DomainCollectorAgent", blackboard=None,
                 message_bus=None, config: Dict[str, Any] = None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self.cfg = config or {}
        self._tool_manager = None
        self._current_task = {}

        # defaults
        self.cfg.setdefault("domain", "unknown")
        self.cfg.setdefault("data_source", "file")
        self.cfg.setdefault("id_field", "id")
        self.cfg.setdefault("label_field", "label")
        self.cfg.setdefault("max_items", 10000)
        self.cfg.setdefault("collector_ns", "collector")
        self.cfg.setdefault("cslo_dir_key", "items")
        self.cfg.setdefault("max_items_param", "max_items")

    # ==================== Tool Manager (lazy) ====================

    def _get_tool_manager(self):
        """懒加载工具管理器。对 file-based 域返回 None（直接用 JSON 加载）。"""
        if self._tool_manager is not None:
            return self._tool_manager
        cls_path = self.cfg.get("tool_manager_class", "")
        if not cls_path:
            return None  # file-based domain
        try:
            mod_path, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            data_dir = self.cfg.get("data_dir", f"./data/{self.cfg['domain']}_raw")
            self._tool_manager = cls(data_dir=data_dir)
            return self._tool_manager
        except Exception as e:
            logger.warning(f"[DomainCollector] 工具管理器加载失败 ({cls_path}): {e}")
            return None

    # ==================== ReAct 接口 ====================

    def get_available_actions(self) -> List[str]:
        return ["load_items", "ensure_cslo_dirs", "validate_items", "save_results", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        res = self._accumulated_results
        if not res.get("items"):
            return {"action": "load_items", "params": {}, "reasoning": "默认: 加载元数据"}
        if not res.get("cslo_checked"):
            return {"action": "ensure_cslo_dirs", "params": {}, "reasoning": "默认: 校验 cs_lo 目录"}
        if not res.get("stats"):
            return {"action": "validate_items", "params": {}, "reasoning": "默认: 验证质量"}
        return {"action": "save_results", "params": {}, "reasoning": "默认: 保存结果"}

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)
        items = self._accumulated_results.get("items", [])
        observation.update({
            "item_count": len(items),
            "cslo_dir": self._cslo_dir(),
            "progress_summary": f"已加载 {len(items)} 个 {self.cfg.get('domain', '')} 条目",
        })
        res = self._accumulated_results
        if not res.get("items"):
            observation["phase_hint"] = "尚未加载数据, 下一步选择 load_items。"
        elif not res.get("cslo_checked"):
            observation["phase_hint"] = f"已加载 {len(items)} 个条目, 下一步 ensure_cslo_dirs 校验目录。"
        elif not res.get("stats"):
            observation["phase_hint"] = "目录已校验, 下一步 validate_items 验证质量。"
        elif not res.get("raw_path"):
            observation["phase_hint"] = "已验证, 下一步必须 save_results 写入 blackboard。"
        else:
            observation["phase_hint"] = "已保存, 选择 done 结束。"
        return observation

    def is_done(self, observation: Dict) -> bool:
        items = self._accumulated_results.get("items", [])
        ns = self.cfg["collector_ns"]
        if items and self.blackboard and self.blackboard.read(ns, "item_count", 0) > 0:
            return True
        return len(self._action_history) >= 6

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})
        action_map = {
            "load_items": self.action_load_items,
            "ensure_cslo_dirs": self.action_ensure_cslo_dirs,
            "validate_items": self.action_validate_items,
            "save_results": self.action_save_results,
        }
        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "采集完成"}
        return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 路径辅助 ====================

    def _cslo_dir(self) -> str:
        key = self.cfg["cslo_dir_key"]
        return self._current_task.get("output_dirs", {}).get(
            key, f"./data/{self.cfg['domain']}_{key}")

    def _raw_dir(self) -> str:
        return self._current_task.get("output_dirs", {}).get(
            "raw", f"./data/{self.cfg['domain']}_raw")

    # ==================== 行动方法 ====================

    def action_load_items(self, params: Dict) -> Dict:
        """加载数据: tool manager (API-based) 或直接 JSON 文件 (file-based)。"""
        tm = self._get_tool_manager()
        max_items = self._current_task.get("max_items", self.cfg["max_items"])

        if tm:
            # API-based: 使用域特定的 tool manager
            param_name = self.cfg.get("max_items_param", "max_items")
            try:
                load_fn = getattr(tm, "load_raw", None)
                if load_fn:
                    items = load_fn(**{param_name: max_items})
                else:
                    return {"status": "error", "message": f"工具管理器无 load_raw 方法"}
            except TypeError:
                # 尝试无参数调用
                items = tm.load_raw()
                items = items[:max_items]
        else:
            # File-based: 直接读 JSON
            data_file = self._current_task.get("data_file") or self.cfg.get("data_file", "")
            if not data_file or not os.path.exists(data_file):
                # 尝试从项目根目录解析相对路径
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent
                data_file = str(project_root / data_file) if data_file else ""
            if not data_file or not os.path.exists(data_file):
                return {"status": "error",
                        "message": f"数据文件不存在: {data_file}。请设置 data_file 或提供 tool_manager_class"}
            with open(data_file, encoding="utf-8") as f:
                raw = json.load(f)
            # 支持嵌套结构: {"items": [...]} 或 [...]
            if isinstance(raw, dict):
                for k in ["items", "data", "games", "movies", "artworks"]:
                    if k in raw and isinstance(raw[k], list) and len(raw[k]) > 0:
                        items = raw[k]
                        break
                else:
                    # 取第一个 list 值
                    for v in raw.values():
                        if isinstance(v, list) and len(v) > 0:
                            items = v
                            break
                    else:
                        items = list(raw.values())
            else:
                items = raw
            items = items[:max_items]

        self._accumulate_result({"items": items})
        logger.info(f"[DomainCollector:{self.cfg['domain']}] 加载 {len(items)} 个条目")
        return {"status": "success", "item_count": len(items)}

    def action_ensure_cslo_dirs(self, params: Dict) -> Dict:
        """校验 cs_lo 目录: 抽样检查 markdown 文件是否存在。"""
        cslo = self._cslo_dir()
        id_field = self.cfg["id_field"]
        items = self._accumulated_results.get("items", [])
        sample = items[:50]
        ok = 0
        for it in sample:
            iid = str(it.get(id_field, ""))
            md_path = os.path.join(cslo, iid, f"{iid}.md")
            if os.path.exists(md_path):
                ok += 1
        self._accumulate_result({"cslo_checked": True, "items_with_markdown": ok})
        logger.info(f"[DomainCollector:{self.cfg['domain']}] cs_lo 校验: {ok}/{len(sample)} 有 md")
        return {"status": "success", "cslo_dir": cslo, "with_md": ok}

    def action_validate_items(self, params: Dict) -> Dict:
        """验证数据质量: 标签覆盖率。"""
        items = self._accumulated_results.get("items", [])
        label_field = self.cfg["label_field"]
        # 提取父字段名 (e.g. "genres[0]" -> "genres", "style" -> "style")
        import re
        m = re.match(r'^(\w+)', label_field)
        key = m.group(1) if m else label_field
        labeled = sum(1 for it in items if it.get(key))
        stats = {
            "total": len(items),
            "labeled": labeled,
            "label_ratio": round(labeled / max(len(items), 1), 3),
        }
        self._accumulate_result({"stats": stats})
        logger.info(f"[DomainCollector:{self.cfg['domain']}] 验证: "
                     f"{labeled}/{len(items)} 有标签 ({stats['label_ratio']})")
        return {"status": "success", "stats": stats}

    def action_save_results(self, params: Dict) -> Dict:
        """保存结果到 blackboard + JSON 文件。"""
        task = self._current_task
        items = self._accumulated_results.get("items", [])
        cslo = self._cslo_dir()
        domain = self.cfg["domain"]
        ns = self.cfg["collector_ns"]

        raw_dir = self._raw_dir()
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"{domain}_items.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

        with_md = self._accumulated_results.get("items_with_markdown", 0)
        if self.blackboard:
            self.blackboard.write(ns, "items", items)
            self.blackboard.write(ns, "item_count", len(items))
            self.blackboard.write(ns, "raw_path", raw_path)
            self.blackboard.write(ns, "cs_lo_dir", cslo)
            self.blackboard.write(ns, "items_with_markdown", with_md)
            self.blackboard.write(ns, "data_dir", cslo)

        logger.info(f"[DomainCollector:{domain}] 保存 {len(items)} 条目 → {raw_path}")
        return {"status": "success", "raw_path": raw_path, "item_count": len(items)}

    # ==================== execute 入口 ====================

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """ReAct 入口 — LLM 驱动的 observe→think→act 循环。"""
        self._current_task = task
        self._upstream_data = upstream_data
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)

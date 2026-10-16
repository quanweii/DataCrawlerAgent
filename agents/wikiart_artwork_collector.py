"""
WikiartArtworkCollectorAgent — WikiArt 画作元数据采集 Agent (仿 PaperCollectorAgent)

ReAct 多智能体: 经 LLM 驱动的 observe→think→act 循环采集元数据
(load_items → ensure_cslo_dirs → validate_items → save_results)。
file 模式下 load_items 从缓存 wikiart_v2_enriched.json 加载; live 模式实时爬取。
id 经 crawlers.wikiart.sanitize_pid 清洗, 与 cs_lo 目录名一致。
每一步由 LLM 决策, _default_action 兜底 — 与 arxiv agent 同款, 非写死流水线。
"""
import os
import json
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message
from crawlers.wikiart import sanitize_pid


class WikiartArtworkCollectorAgent(ReActAgent):
    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="WikiartArtworkCollectorAgent", blackboard=None,
                 message_bus=None, wikiart_tools=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self.wikiart_tools = wikiart_tools
        self._current_task = {}

    def get_available_actions(self) -> List[str]:
        return ["load_items", "ensure_cslo_dirs", "validate_items", "save_results", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        results = self._accumulated_results
        if not results.get("items"):
            return {"action": "load_items", "params": {}, "reasoning": "默认: 加载画作元数据"}
        elif not results.get("cslo_checked"):
            return {"action": "ensure_cslo_dirs", "params": {}, "reasoning": "默认: 校验 cs_lo 目录"}
        elif not results.get("stats"):
            return {"action": "validate_items", "params": {}, "reasoning": "默认: 验证质量"}
        return {"action": "save_results", "params": {}, "reasoning": "默认: 保存结果"}

    def observe(self, task, blackboard, messages):
        observation = super().observe(task, blackboard, messages)
        items = self._accumulated_results.get("items", [])
        observation.update({"item_count": len(items), "cslo_dir": self._cslo_dir(),
                            "progress_summary": f"已加载 {len(items)} 幅画作"})
        # phase_hint 引导 LLM 决策下一步 (同 PaperCollectorAgent 模式, 非写死控制流)
        res = self._accumulated_results
        if not res.get("items"):
            observation["phase_hint"] = "尚未加载画作, 下一步选择 load_items。"
        elif not res.get("cslo_checked"):
            observation["phase_hint"] = f"已加载 {len(items)} 幅画作, 下一步 ensure_cslo_dirs 校验目录。"
        elif not res.get("stats"):
            observation["phase_hint"] = "目录已校验, 下一步 validate_items 验证质量。"
        elif not res.get("raw_path"):
            observation["phase_hint"] = "已验证, 下一步必须 save_results 写入 blackboard。"
        else:
            observation["phase_hint"] = "已保存, 选择 done 结束。"
        return observation

    def is_done(self, observation: Dict) -> bool:
        items = self._accumulated_results.get("items", [])
        if items and self.blackboard and self.blackboard.read("wikiart_collector", "item_count", 0) > 0:
            return True
        return len(self._action_history) >= 6

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})
        m = {"load_items": self.action_load_items, "ensure_cslo_dirs": self.action_ensure_cslo_dirs,
             "validate_items": self.action_validate_items, "save_results": self.action_save_results}
        h = m.get(action)
        if h:
            return h(params)
        elif action == "done":
            return {"status": "success", "message": "采集完成"}
        return {"status": "error", "message": f"未知行动: {action}"}

    def _cslo_dir(self) -> str:
        return self._current_task.get("output_dirs", {}).get("artworks", "./data/wikiart_artworks_v2")

    def action_load_items(self, params: Dict) -> Dict:
        if self.wikiart_tools is None:
            try:
                from tools.wikiart_tools import WikiartToolManager
                self.wikiart_tools = WikiartToolManager()
            except Exception as e:
                return {"status": "error", "message": f"WikiArt 工具初始化失败: {e}"}
        source = self._current_task.get("data_source", "file")
        if source in ("file", "cache") or self._current_task.get("load_from_files"):
            items = self.wikiart_tools.load_raw(max_artworks=self._current_task.get("max_artworks"))
        else:
            items = self.wikiart_tools.crawl_live(self._current_task.get("max_artworks", 22000))
        self._accumulate_result({"items": items})
        return {"status": "success", "item_count": len(items)}

    def action_ensure_cslo_dirs(self, params: Dict) -> Dict:
        cslo = self._cslo_dir()
        items = self._accumulated_results.get("items", [])
        sample = items[:50]
        ok = sum(1 for it in sample
                 if os.path.exists(os.path.join(cslo, sanitize_pid(it.get("painting_id", "")),
                                                f"{sanitize_pid(it.get('painting_id', ''))}.md")))
        self._accumulate_result({"cslo_checked": True, "items_with_markdown": ok})
        return {"status": "success", "cslo_dir": cslo, "with_md": ok}

    def action_validate_items(self, params: Dict) -> Dict:
        items = self._accumulated_results.get("items", [])
        labeled = sum(1 for it in items if it.get("style"))
        stats = {"total": len(items), "labeled": labeled,
                 "label_ratio": round(labeled / max(len(items), 1), 3)}
        self._accumulate_result({"stats": stats})
        return {"status": "success", "stats": stats}

    def action_save_results(self, params: Dict) -> Dict:
        task = self._current_task
        items = self._accumulated_results.get("items", [])
        cslo = self._cslo_dir()
        raw_dir = task.get("output_dirs", {}).get("raw", "./data/wikiart_raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, "wikiart_items.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        with_md = self._accumulated_results.get("items_with_markdown", 0)
        if self.blackboard:
            self.blackboard.write("wikiart_collector", "items", items)
            self.blackboard.write("wikiart_collector", "item_count", len(items))
            self.blackboard.write("wikiart_collector", "raw_path", raw_path)
            self.blackboard.write("wikiart_collector", "cs_lo_dir", cslo)
            self.blackboard.write("wikiart_collector", "items_with_markdown", with_md)
        return {"status": "success", "raw_path": raw_path, "item_count": len(items)}

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """ReAct 入口 — 经 LLM 驱动的 observe→think→act 循环采集 (同 arxiv PaperCollectorAgent)。
        每一步由 think() 的 LLM 决策; observe() 的 phase_hint 引导决策;
        LLM 失败时 _default_action 兜底 (load_items→ensure_cslo_dirs→validate_items→save_results)。"""
        self._current_task = task
        self._upstream_data = upstream_data
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)

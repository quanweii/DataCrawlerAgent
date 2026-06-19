"""
SteamGameCollectorAgent — Steam 游戏元数据采集 Agent (仿 PaperCollectorAgent)

ReAct 多智能体: 经 LLM 驱动的 observe→think→act 循环采集元数据
(load_items → ensure_cslo_dirs → validate_items → save_results)。
file 模式下 load_items 从缓存 steam_games_10k.json 加载已爬取元数据; live 模式实时爬取。
每一步由 LLM 决策, _default_action 兜底 — 与 arxiv agent 同款, 非写死流水线。
"""
import os
import json
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class SteamGameCollectorAgent(ReActAgent):
    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="SteamGameCollectorAgent", blackboard=None,
                 message_bus=None, steam_tools=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self.steam_tools = steam_tools
        self._current_task = {}

    # ==================== ReAct 接口 ====================

    def get_available_actions(self) -> List[str]:
        return ["load_items", "ensure_cslo_dirs", "validate_items",
                "save_results", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        results = self._accumulated_results
        items = results.get("items", [])
        if not items:
            return {"action": "load_items", "params": {},
                    "reasoning": "默认: 先加载游戏元数据"}
        elif not results.get("cslo_checked"):
            return {"action": "ensure_cslo_dirs", "params": {},
                    "reasoning": "默认: 校验 cs_lo 目录"}
        elif not results.get("stats"):
            return {"action": "validate_items", "params": {},
                    "reasoning": "默认: 验证游戏质量"}
        else:
            return {"action": "save_results", "params": {},
                    "reasoning": "默认: 保存结果"}

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)
        items = self._accumulated_results.get("items", [])
        observation.update({
            "item_count": len(items),
            "cslo_dir": self._cslo_dir(),
            "progress_summary": f"已加载 {len(items)} 个 Steam 游戏",
        })
        # phase_hint 引导 LLM 决策下一步 (同 PaperCollectorAgent 模式, 非写死控制流)
        res = self._accumulated_results
        if not res.get("items"):
            observation["phase_hint"] = "尚未加载游戏, 下一步选择 load_items。"
        elif not res.get("cslo_checked"):
            observation["phase_hint"] = f"已加载 {len(items)} 个游戏, 下一步 ensure_cslo_dirs 校验目录。"
        elif not res.get("stats"):
            observation["phase_hint"] = "目录已校验, 下一步 validate_items 验证质量。"
        elif not res.get("raw_path"):
            observation["phase_hint"] = "已验证, 下一步必须 save_results 写入 blackboard。"
        else:
            observation["phase_hint"] = "已保存, 选择 done 结束。"
        return observation

    def is_done(self, observation: Dict) -> bool:
        items = self._accumulated_results.get("items", [])
        if items and self.blackboard:
            if self.blackboard.read("steam_collector", "item_count", 0) > 0:
                return True
        if len(self._action_history) >= 6:
            return True
        return False

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

    # ==================== 行动方法 ====================

    def _cslo_dir(self) -> str:
        return self._current_task.get("output_dirs", {}).get("games", "./data/steam_games")

    def action_load_items(self, params: Dict) -> Dict:
        if self.steam_tools is None:
            try:
                from tools.steam_tools import SteamToolManager
                self.steam_tools = SteamToolManager()
            except Exception as e:
                return {"status": "error", "message": f"Steam 工具初始化失败: {e}"}
        source = self._current_task.get("data_source", "file")
        if source in ("file", "cache") or self._current_task.get("load_from_files"):
            items = self.steam_tools.load_raw(max_games=self._current_task.get("max_games"))
        else:
            items = self.steam_tools.crawl_live(self._current_task.get("max_games", 10000))
        self._accumulate_result({"items": items})
        return {"status": "success", "message": f"加载 {len(items)} 游戏",
                "item_count": len(items)}

    def action_ensure_cslo_dirs(self, params: Dict) -> Dict:
        cslo = self._cslo_dir()
        if not os.path.isdir(cslo):
            return {"status": "warning", "message": f"cs_lo 目录不存在: {cslo}"}
        # 抽样校验目录结构
        items = self._accumulated_results.get("items", [])
        sample = items[:50]
        ok = sum(1 for it in sample
                 if os.path.exists(os.path.join(cslo, str(it.get("appid")), f"{it.get('appid')}.md")))
        with_md = ok
        self._accumulate_result({"cslo_checked": True, "items_with_markdown": with_md})
        return {"status": "success", "message": f"cs_lo 校验完成, 抽样 {ok}/{len(sample)} 有 md",
                "cslo_dir": cslo}

    def action_validate_items(self, params: Dict) -> Dict:
        items = self._accumulated_results.get("items", [])
        labeled = sum(1 for it in items if it.get("genres"))
        stats = {"total": len(items), "labeled": labeled,
                 "label_ratio": round(labeled / max(len(items), 1), 3)}
        self._accumulate_result({"stats": stats})
        return {"status": "success", "stats": stats}

    def action_save_results(self, params: Dict) -> Dict:
        task = self._current_task
        items = self._accumulated_results.get("items", [])
        cslo = self._cslo_dir()
        raw_dir = task.get("output_dirs", {}).get("raw", "./data/steam_raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, "steam_items.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

        with_md = self._accumulated_results.get("items_with_markdown", 0)
        if self.blackboard:
            self.blackboard.write("steam_collector", "items", items)
            self.blackboard.write("steam_collector", "item_count", len(items))
            self.blackboard.write("steam_collector", "raw_path", raw_path)
            self.blackboard.write("steam_collector", "cs_lo_dir", cslo)
            self.blackboard.write("steam_collector", "items_with_markdown", with_md)
        return {"status": "success", "message": f"保存 {len(items)} 游戏",
                "raw_path": raw_path, "item_count": len(items)}

    # ==================== execute 入口 ====================

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

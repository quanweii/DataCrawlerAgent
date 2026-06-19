"""
SubredditCollectorAgent — Reddit 子版块发现与帖子采集 Agent
从指定 subreddit 采集帖子元数据，支持多种排序方式
"""
import os
import json
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class SubredditCollectorAgent(ReActAgent):
    """
    Reddit Subreddit 采集 Agent

    可用行动:
    - discover_subreddits: 搜索/验证目标 subreddit
    - fetch_posts: 采集帖子元数据
    - validate_posts: 验证采集结果
    - expand_search: 扩展搜索更多 subreddit
    - save_results: 保存采集结果
    - done: 完成
    """

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="SubredditCollectorAgent", blackboard=None,
                 message_bus=None, reddit_tools=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self.reddit_tools = reddit_tools
        self._current_task = {}

    def get_available_actions(self) -> List[str]:
        return [
            "discover_subreddits",
            "fetch_posts",
            "validate_posts",
            "expand_search",
            "save_results",
            "done"
        ]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动"""
        results = self._accumulated_results
        subreddits = results.get("subreddits", [])
        posts = results.get("posts", [])

        if not subreddits:
            return {"action": "discover_subreddits", "params": {},
                    "reasoning": "默认: 先发现subreddit"}
        elif not posts:
            return {"action": "fetch_posts", "params": {},
                    "reasoning": "默认: 已有subreddit, 采集帖子"}
        elif not results.get("stats"):
            return {"action": "validate_posts", "params": {},
                    "reasoning": "默认: 验证帖子质量"}
        else:
            return {"action": "save_results", "params": {},
                    "reasoning": "默认: 保存结果"}

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        subreddits = self._accumulated_results.get("subreddits", [])
        posts = self._accumulated_results.get("posts", [])
        target_count = task.get("posts_per_subreddit", 100)

        observation.update({
            "subreddit_count": len(subreddits),
            "total_posts": len(posts),
            "target_per_subreddit": target_count,
            "progress_summary": (
                f"已发现 {len(subreddits)} 个 subreddit, "
                f"共 {len(posts)} 帖子"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        total_posts = observation.get("total_posts", 0)
        subreddit_count = observation.get("subreddit_count", 0)

        # 只有当 save_results 已执行（blackboard 有数据）时才完成
        if total_posts > 0 and self.blackboard:
            saved = self.blackboard.read("subreddit_collector", "post_count", 0)
            if saved > 0:
                return True

        # 安全阀：执行了足够多的行动
        if len(self._action_history) >= 8:
            return True

        return False

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "discover_subreddits": self.action_discover_subreddits,
            "fetch_posts": self.action_fetch_posts,
            "validate_posts": self.action_validate_posts,
            "expand_search": self.action_expand_search,
            "save_results": self.action_save_results,
        }

        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "任务完成"}
        else:
            return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 行动实现 ====================

    def action_discover_subreddits(self, params: Dict) -> Dict:
        """发现并验证目标 subreddit"""
        task = self._current_task
        target_subreddits = task.get("subreddits", [])

        if not target_subreddits:
            # 如果没有指定，根据关键词搜索
            keyword = task.get("subreddit_keyword", "machine learning")
            subreddits = self.reddit_tools.search_subreddits(keyword, limit=10)
            target_subreddits = [s["name"] for s in subreddits]

        # 验证每个 subreddit 并获取信息
        validated = []
        for name in target_subreddits:
            info = self.reddit_tools.fetch_subreddit_info(name)
            if info:
                validated.append(info)

        self._accumulate_result({"subreddits": validated})
        logger.info(
            f"[SubredditCollector] 验证 {len(validated)}/{len(target_subreddits)} "
            f"个 subreddit"
        )

        return {
            "status": "success",
            "message": f"验证 {len(validated)} 个 subreddit",
            "subreddit_count": len(validated)
        }

    def action_fetch_posts(self, params: Dict) -> Dict:
        """从已发现的 subreddit 采集帖子"""
        task = self._current_task
        subreddits = self._accumulated_results.get("subreddits", [])

        if not subreddits:
            # 尝试从任务配置获取
            subreddit_names = task.get("subreddits", [])
            subreddits = [{"name": n} for n in subreddit_names]

        sort = task.get("sort", "hot")
        limit = task.get("posts_per_subreddit", 100)
        time_filter = task.get("time_filter", "year")

        all_posts = []
        for sub_info in subreddits:
            name = sub_info.get("name", "")
            if not name:
                continue

            posts = self.reddit_tools.fetch_subreddit_posts(
                subreddit_name=name,
                sort=sort,
                limit=limit,
                time_filter=time_filter
            )

            all_posts.extend(posts)
            logger.info(
                f"[SubredditCollector] r/{name}: {len(posts)} 帖子"
            )

        # 去重（跨 subreddit 可能有 crosspost 重复）
        seen_ids = set()
        unique_posts = []
        for p in all_posts:
            if p["post_id"] not in seen_ids:
                seen_ids.add(p["post_id"])
                unique_posts.append(p)

        self._accumulate_result({"posts": unique_posts})

        return {
            "status": "success",
            "message": f"采集 {len(unique_posts)} 帖子 (去重前 {len(all_posts)})",
            "post_count": len(unique_posts)
        }

    def action_validate_posts(self, params: Dict) -> Dict:
        """验证采集的帖子数据质量"""
        posts = self._accumulated_results.get("posts", [])

        issues = []
        stats = {
            "total": len(posts),
            "has_text": 0,
            "has_image": 0,
            "has_author": 0,
            "has_comments": 0,
            "deleted_author": 0,
        }

        for post in posts:
            if post.get("text_length", 0) > 0:
                stats["has_text"] += 1
            if post.get("has_image", False):
                stats["has_image"] += 1
            if post.get("author", "") not in ("", "[deleted]"):
                stats["has_author"] += 1
            else:
                stats["deleted_author"] += 1
            if post.get("num_comments", 0) > 0:
                stats["has_comments"] += 1

        # 检查质量问题
        if stats["deleted_author"] > stats["total"] * 0.3:
            issues.append(
                f"过多删除用户 ({stats['deleted_author']}/{stats['total']})"
            )

        if stats["has_image"] < stats["total"] * 0.1:
            issues.append(
                f"图片帖比例过低 ({stats['has_image']}/{stats['total']})"
            )

        result = {
            "status": "success",
            "message": f"验证完成: {stats}",
            "stats": stats,
            "issues": issues,
        }

        # 如果有质量问题，通知下游
        if issues:
            result["_messages"] = [{
                "recipient": "broadcast",
                "msg_type": "quality_issue",
                "content": {"source": "SubredditCollectorAgent", "issues": issues},
                "priority": 1
            }]

        return result

    def action_expand_search(self, params: Dict) -> Dict:
        """扩展搜索更多相关 subreddit"""
        keyword = params.get("keyword", "artificial intelligence deep learning")
        new_subs = self.reddit_tools.search_subreddits(keyword, limit=5)

        existing_names = {
            s.get("name") for s in self._accumulated_results.get("subreddits", [])
        }
        new_validated = []
        for sub in new_subs:
            if sub["name"] not in existing_names:
                info = self.reddit_tools.fetch_subreddit_info(sub["name"])
                if info:
                    new_validated.append(info)

        if new_validated:
            self._accumulate_result({"subreddits": new_validated})

        return {
            "status": "success",
            "message": f"扩展搜索发现 {len(new_validated)} 个新 subreddit",
            "new_count": len(new_validated)
        }

    def action_save_results(self, params: Dict) -> Dict:
        """保存采集结果到文件"""
        task = self._current_task
        posts = self._accumulated_results.get("posts", [])
        subreddits = self._accumulated_results.get("subreddits", [])

        output_dir = task.get("output_dirs", {}).get("raw", "./data/reddit_raw")
        os.makedirs(output_dir, exist_ok=True)

        # 保存帖子
        posts_path = os.path.join(output_dir, "posts.json")
        with open(posts_path, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)

        # 保存 subreddit 信息
        subs_path = os.path.join(output_dir, "subreddits.json")
        with open(subs_path, 'w', encoding='utf-8') as f:
            json.dump(subreddits, f, ensure_ascii=False, indent=2)

        # 写入 blackboard 供下游 Agent 读取
        if self.blackboard:
            self.blackboard.write("subreddit_collector", "posts", posts)
            self.blackboard.write("subreddit_collector", "post_count", len(posts))
            self.blackboard.write("subreddit_collector", "subreddits", subreddits)
            self.blackboard.write("subreddit_collector", "subreddit_count", len(subreddits))

        return {
            "status": "success",
            "message": f"保存 {len(posts)} 帖子, {len(subreddits)} subreddit",
            "posts_path": posts_path,
            "subreddits_path": subs_path,
        }

    def _try_load_from_files(self) -> bool:
        """
        尝试从预加载的数据文件加载（Pushshift 或其他来源）
        如果 data_source 为 "file" 且存在预加载文件，直接加载并写入 blackboard

        Returns:
            True 如果成功从文件加载了数据
        """
        task = self._current_task
        output_dir = task.get("output_dirs", {}).get("raw", "./data/reddit_raw")

        posts_path = os.path.join(output_dir, "posts.json")
        subs_path = os.path.join(output_dir, "subreddits.json")

        if not os.path.exists(posts_path):
            return False

        try:
            with open(posts_path, 'r', encoding='utf-8') as f:
                posts = json.load(f)

            subreddits = []
            if os.path.exists(subs_path):
                with open(subs_path, 'r', encoding='utf-8') as f:
                    subreddits = json.load(f)
            else:
                # 从帖子中推断 subreddit
                seen = set()
                for post in posts:
                    sub = post.get("subreddit", "")
                    if sub and sub not in seen:
                        seen.add(sub)
                        subreddits.append({"name": sub, "post_count": 0})
                # 统计每个 subreddit 的帖子数
                sub_counts = {}
                for post in posts:
                    sub = post.get("subreddit", "")
                    sub_counts[sub] = sub_counts.get(sub, 0) + 1
                for s in subreddits:
                    s["post_count"] = sub_counts.get(s["name"], 0)

            self._accumulate_result({"posts": posts, "subreddits": subreddits})

            # 写入 blackboard
            if self.blackboard:
                self.blackboard.write("subreddit_collector", "posts", posts)
                self.blackboard.write("subreddit_collector", "post_count", len(posts))
                self.blackboard.write("subreddit_collector", "subreddits", subreddits)
                self.blackboard.write("subreddit_collector", "subreddit_count", len(subreddits))

            logger.info(
                f"[SubredditCollector] 从文件加载: "
                f"{len(posts)} 帖子, {len(subreddits)} subreddit"
            )
            return True

        except Exception as e:
            logger.warning(f"[SubredditCollector] 从文件加载失败: {e}")
            return False

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """执行入口"""
        self._current_task = task
        self._accumulated_results = {}

        # 优先尝试从预加载文件读取（Pushshift / 其他数据源）
        if task.get("data_source") == "file" or task.get("load_from_files"):
            if self._try_load_from_files():
                return {
                    "status": "success",
                    "message": (
                        f"从预加载文件读取: "
                        f"{len(self._accumulated_results.get('posts', []))} 帖子"
                    ),
                    "post_count": len(self._accumulated_results.get("posts", [])),
                    "data_source": "file",
                }

        # 如果没有 reddit_tools，尝试创建
        if self.reddit_tools is None:
            try:
                from tools.reddit_tools import RedditToolManager
                self.reddit_tools = RedditToolManager()
            except Exception as e:
                logger.error(f"[SubredditCollector] 初始化 Reddit 工具失败: {e}")
                return {
                    "status": "error",
                    "message": f"Reddit 工具初始化失败: {e}"
                }

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )

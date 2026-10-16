"""
PostCollectorAgent — Reddit 评论采集与图片下载 Agent
从 SubredditCollectorAgent 采集的帖子中，进一步采集评论树和下载图片
"""
import os
import json
from typing import Dict, Any, List
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class PostCollectorAgent(ReActAgent):
    """
    Reddit 帖子深度采集 Agent

    可用行动:
    - collect_comments: 采集评论树
    - download_images: 下载帖子图片
    - collect_user_metadata: 收集用户元数据
    - validate_collection: 验证采集结果
    - save_results: 保存结果
    - done: 完成
    """

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="PostCollectorAgent", blackboard=None,
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
            "collect_comments",
            "download_images",
            "collect_user_metadata",
            "validate_collection",
            "save_results",
            "done"
        ]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动"""
        results = self._accumulated_results
        total_comments = results.get("total_comments", 0)
        images_attempted = "image_stats" in results

        if total_comments == 0:
            return {"action": "collect_comments", "params": {},
                    "reasoning": "默认: 先采集评论"}
        elif not images_attempted:
            return {"action": "download_images", "params": {},
                    "reasoning": "默认: 下载图片"}
        elif not results.get("user_metadata"):
            return {"action": "collect_user_metadata", "params": {},
                    "reasoning": "默认: 收集用户信息"}
        else:
            return {"action": "save_results", "params": {},
                    "reasoning": "默认: 保存结果"}

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        posts_with_comments = self._accumulated_results.get("posts_with_comments", 0)
        total_comments = self._accumulated_results.get("total_comments", 0)
        images_downloaded = self._accumulated_results.get("images_downloaded", 0)

        observation.update({
            "posts_with_comments": posts_with_comments,
            "total_comments": total_comments,
            "images_downloaded": images_downloaded,
            "progress_summary": (
                f"评论: {total_comments} ({posts_with_comments} 帖子), "
                f"图片: {images_downloaded}"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        total_comments = observation.get("total_comments", 0)

        # 只有当 save_results 已执行时才完成
        if total_comments > 0 and self.blackboard:
            saved = self.blackboard.read("post_collector", "total_comments", 0)
            if saved > 0:
                return True

        if len(self._action_history) >= 8:
            return True
        return False

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "collect_comments": self.action_collect_comments,
            "download_images": self.action_download_images,
            "collect_user_metadata": self.action_collect_user_metadata,
            "validate_collection": self.action_validate_collection,
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

    def action_collect_comments(self, params: Dict) -> Dict:
        """采集帖子评论"""
        task = self._current_task

        # 从 blackboard 或 task 获取帖子列表
        posts = self._get_posts()

        if not posts:
            return {"status": "error", "message": "没有可用的帖子数据"}

        max_depth = task.get("comment_depth", 3)
        limit_per_post = task.get("comments_per_post", 25)

        post_ids = [p["post_id"] for p in posts]
        comments_map = self.reddit_tools.fetch_batch_post_comments(
            post_ids=post_ids,
            max_depth=max_depth,
            limit=limit_per_post,
            max_workers=2  # 保守并发，避免限流
        )

        # 统计
        total_comments = sum(len(c) for c in comments_map.values())
        posts_with_comments = sum(1 for c in comments_map.values() if len(c) > 0)

        # 收集所有用户
        all_users = set()
        for comments in comments_map.values():
            for c in comments:
                if c.get("author", "") not in ("", "[deleted]"):
                    all_users.add(c["author"])

        # 帖子作者
        for post in posts:
            author = post.get("author", "")
            if author not in ("", "[deleted]"):
                all_users.add(author)

        self._accumulate_result({
            "comments_map": comments_map,
            "total_comments": total_comments,
            "posts_with_comments": posts_with_comments,
            "users": list(all_users),
            "user_count": len(all_users),
        })

        logger.info(
            f"[PostCollector] 评论采集: {total_comments} 评论, "
            f"{posts_with_comments} 帖子有评论, {len(all_users)} 用户"
        )

        return {
            "status": "success",
            "message": f"采集 {total_comments} 评论",
            "total_comments": total_comments,
            "posts_with_comments": posts_with_comments,
        }

    def action_download_images(self, params: Dict) -> Dict:
        """下载帖子图片"""
        task = self._current_task
        posts = self._get_posts()

        if not posts:
            return {"status": "error", "message": "没有帖子数据"}

        image_dir = task.get("output_dirs", {}).get("images", "./data/reddit_raw/images")
        image_stats = self.reddit_tools.download_reddit_images(
            posts,
            output_dir=image_dir,
            max_workers=5
        )

        self._accumulate_result({
            "images_downloaded": image_stats.get("downloaded", 0),
            "images_failed": image_stats.get("failed", 0),
            "image_stats": image_stats,
        })

        return {
            "status": "success",
            "message": (
                f"图片下载: {image_stats.get('downloaded', 0)} 成功, "
                f"{image_stats.get('failed', 0)} 失败"
            ),
            **image_stats
        }

    def action_collect_user_metadata(self, params: Dict) -> Dict:
        """采集用户元数据（karma 等）"""
        users = self._accumulated_results.get("users", [])

        if not users:
            return {"status": "success", "message": "无用户需采集"}

        # 限制采集数量（避免 API 过载）
        task = self._current_task
        max_users = task.get("max_user_profiles", 500)
        target_users = users[:max_users]

        user_metadata = {}
        for username in target_users:
            info = self.reddit_tools.fetch_user_info(username)
            if info:
                user_metadata[username] = info

        self._accumulate_result({"user_metadata": user_metadata})

        logger.info(
            f"[PostCollector] 采集 {len(user_metadata)}/{len(target_users)} 用户信息"
        )

        return {
            "status": "success",
            "message": f"采集 {len(user_metadata)} 用户元数据",
            "user_count": len(user_metadata),
        }

    def action_validate_collection(self, params: Dict) -> Dict:
        """验证采集完整性"""
        posts = self._get_posts()
        comments_map = self._accumulated_results.get("comments_map", {})
        total_comments = self._accumulated_results.get("total_comments", 0)

        stats = {
            "total_posts": len(posts),
            "posts_with_comments": sum(1 for c in comments_map.values() if c),
            "total_comments": total_comments,
            "unique_users": self._accumulated_results.get("user_count", 0),
            "images_downloaded": self._accumulated_results.get("images_downloaded", 0),
        }

        issues = []
        if stats["posts_with_comments"] < stats["total_posts"] * 0.3:
            issues.append("评论覆盖率过低")

        return {
            "status": "success",
            "message": f"验证统计: {stats}",
            "stats": stats,
            "issues": issues,
        }

    def action_save_results(self, params: Dict) -> Dict:
        """保存所有采集结果"""
        task = self._current_task
        output_dir = task.get("output_dirs", {}).get("raw", "./data/reddit_raw")
        os.makedirs(output_dir, exist_ok=True)

        comments_map = self._accumulated_results.get("comments_map", {})
        user_metadata = self._accumulated_results.get("user_metadata", {})

        # 保存评论
        comments_path = os.path.join(output_dir, "comments.json")
        with open(comments_path, 'w', encoding='utf-8') as f:
            json.dump(comments_map, f, ensure_ascii=False, indent=2)

        # 保存用户信息
        users_path = os.path.join(output_dir, "users.json")
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(user_metadata, f, ensure_ascii=False, indent=2)

        # 写入 blackboard
        if self.blackboard:
            self.blackboard.write("post_collector", "comments_map", comments_map)
            self.blackboard.write("post_collector", "total_comments",
                                  self._accumulated_results.get("total_comments", 0))
            self.blackboard.write("post_collector", "user_metadata", user_metadata)
            self.blackboard.write("post_collector", "user_count",
                                  self._accumulated_results.get("user_count", 0))

        return {
            "status": "success",
            "message": "结果已保存",
            "comments_path": comments_path,
            "users_path": users_path,
        }

    # ==================== 辅助方法 ====================

    def _get_posts(self) -> List[Dict]:
        """获取帖子列表（从 blackboard 或累积结果）"""
        posts = self._accumulated_results.get("posts", [])
        if not posts and self.blackboard:
            posts = self.blackboard.read("subreddit_collector", "posts", [])
        if not posts:
            posts = self._current_task.get("posts", [])
        return posts

    def _try_load_from_files(self) -> bool:
        """
        尝试从预加载的数据文件加载评论和用户数据（Pushshift 或其他来源）

        Returns:
            True 如果成功从文件加载了数据
        """
        task = self._current_task
        output_dir = task.get("output_dirs", {}).get("raw", "./data/reddit_raw")

        comments_path = os.path.join(output_dir, "comments.json")
        users_path = os.path.join(output_dir, "users.json")

        if not os.path.exists(comments_path):
            return False

        try:
            with open(comments_path, 'r', encoding='utf-8') as f:
                comments_map = json.load(f)

            user_metadata = {}
            if os.path.exists(users_path):
                with open(users_path, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                    # users 可能是 list 或 dict
                    if isinstance(users_data, list):
                        for u in users_data:
                            username = u.get("username", "")
                            if username:
                                user_metadata[username] = u
                    elif isinstance(users_data, dict):
                        user_metadata = users_data

            total_comments = sum(len(c) for c in comments_map.values())
            posts_with_comments = sum(1 for c in comments_map.values() if c)

            self._accumulate_result({
                "comments_map": comments_map,
                "total_comments": total_comments,
                "posts_with_comments": posts_with_comments,
                "user_metadata": user_metadata,
                "user_count": len(user_metadata),
                "images_downloaded": 0,
                "image_stats": {"downloaded": 0, "skipped": 0, "failed": 0},
            })

            # 写入 blackboard
            if self.blackboard:
                self.blackboard.write("post_collector", "comments_map", comments_map)
                self.blackboard.write("post_collector", "total_comments", total_comments)
                self.blackboard.write("post_collector", "user_metadata", user_metadata)
                self.blackboard.write("post_collector", "user_count", len(user_metadata))

            logger.info(
                f"[PostCollector] 从文件加载: "
                f"{total_comments} 评论, {len(user_metadata)} 用户"
            )
            return True

        except Exception as e:
            logger.warning(f"[PostCollector] 从文件加载失败: {e}")
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
                        f"{self._accumulated_results.get('total_comments', 0)} 评论"
                    ),
                    "total_comments": self._accumulated_results.get("total_comments", 0),
                    "data_source": "file",
                }

        if self.reddit_tools is None:
            try:
                from tools.reddit_tools import RedditToolManager
                self.reddit_tools = RedditToolManager()
            except Exception as e:
                logger.error(f"[PostCollector] 初始化 Reddit 工具失败: {e}")
                return {"status": "error", "message": f"Reddit 工具初始化失败: {e}"}

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )

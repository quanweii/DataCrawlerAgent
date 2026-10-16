"""
SocialGraphBuilderAgent — Reddit 社交图构建 Agent
从采集的帖子和评论中构建 User-Post-Comment-Subreddit 异构图
或 MAGB 兼容的同构图（post 节点 + co-commenting 边）
输出多类型边 CSV 文件
"""
import os
import csv
import json
from typing import Dict, Any, List, Set
from collections import defaultdict
from itertools import combinations
from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus, Message


class SocialGraphBuilderAgent(ReActAgent):
    """
    Reddit 社交图构建 Agent — 支持双模建图

    graph_mode="heterogeneous" (默认):
      - author_of: user→post, user→comment
      - reply_to: comment→post, comment→comment
      - belongs_to: post→subreddit
      - crosspost: post→post

    graph_mode="homogeneous" (MAGB兼容):
      - co_comment: post→post (同一用户在两个帖子下都评论过 → 无向边)
      - 标签 = subreddit 分类

    可用行动:
      异构模式: collect_entities, build_author_edges, build_reply_edges,
               build_belongsto_edges, build_crosspost_edges, save_edges, validate_graph
      同构模式: collect_posts, build_cocomment_edges, save_edges, validate_graph
      通用: done
    """

    max_iterations = 10

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="SocialGraphBuilderAgent", blackboard=None,
                 message_bus=None):
        super().__init__(
            client=client, system_prompt=system_prompt,
            mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
            blackboard=blackboard, message_bus=message_bus
        )
        self._current_task = {}
        self.graph_mode = "heterogeneous"
        # 累积的边
        self._edges: Dict[str, List[Dict]] = defaultdict(list)
        self._entities: Dict[str, Set[str]] = defaultdict(set)
        # 同构模式：用户→帖子映射 和 标签映射
        self._user_to_posts: Dict[str, Set[str]] = defaultdict(set)
        self._label_map: Dict[str, str] = {}

    def get_available_actions(self) -> List[str]:
        return [
            "collect_entities",
            "collect_posts",
            "build_author_edges",
            "build_reply_edges",
            "build_belongsto_edges",
            "build_crosspost_edges",
            "build_cocomment_edges",
            "save_edges",
            "validate_graph",
            "done"
        ]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        """按阶段推进的默认行动 — 根据 graph_mode 分叉"""
        if self.graph_mode == "homogeneous":
            return self._default_action_homogeneous(observation)
        else:
            return self._default_action_heterogeneous(observation)

    def _default_action_heterogeneous(self, observation: Dict) -> Dict[str, Any]:
        """异构模式的默认行动序列"""
        edge_counts = {k: len(v) for k, v in self._edges.items()}
        entity_counts = {k: len(v) for k, v in self._entities.items()}

        if not entity_counts:
            return {"action": "collect_entities", "params": {},
                    "reasoning": "默认: 先收集实体"}
        elif "author_of" not in edge_counts:
            return {"action": "build_author_edges", "params": {},
                    "reasoning": "默认: 构建author_of边"}
        elif "reply_to" not in edge_counts:
            return {"action": "build_reply_edges", "params": {},
                    "reasoning": "默认: 构建reply_to边"}
        elif "belongs_to" not in edge_counts:
            return {"action": "build_belongsto_edges", "params": {},
                    "reasoning": "默认: 构建belongs_to边"}
        elif "crosspost" not in edge_counts:
            return {"action": "build_crosspost_edges", "params": {},
                    "reasoning": "默认: 构建crosspost边"}
        elif not any(h["action"] == "save_edges" and h["result_status"] == "success"
                     for h in self._action_history):
            return {"action": "save_edges", "params": {},
                    "reasoning": "默认: 保存边文件"}
        else:
            return {"action": "validate_graph", "params": {},
                    "reasoning": "默认: 验证图"}

    def _default_action_homogeneous(self, observation: Dict) -> Dict[str, Any]:
        """同构模式的默认行动序列"""
        edge_counts = {k: len(v) for k, v in self._edges.items()}
        has_posts = len(self._entities.get("post", set())) > 0

        if not has_posts:
            return {"action": "collect_posts", "params": {},
                    "reasoning": "默认: 收集帖子节点"}
        elif not edge_counts.get("co_comment") and "co_comment" not in edge_counts:
            return {"action": "build_cocomment_edges", "params": {},
                    "reasoning": "默认: 构建co-commenting边"}
        elif not any(h["action"] == "save_edges" and h["result_status"] == "success"
                     for h in self._action_history):
            return {"action": "save_edges", "params": {},
                    "reasoning": "默认: 保存边文件"}
        else:
            return {"action": "validate_graph", "params": {},
                    "reasoning": "默认: 验证图"}

    def observe(self, task: Dict, blackboard: Blackboard,
                messages: List[Message]) -> Dict[str, Any]:
        observation = super().observe(task, blackboard, messages)

        edge_counts = {k: len(v) for k, v in self._edges.items()}
        entity_counts = {k: len(v) for k, v in self._entities.items()}

        observation.update({
            "graph_mode": self.graph_mode,
            "edge_counts": edge_counts,
            "entity_counts": entity_counts,
            "total_edges": sum(edge_counts.values()),
            "progress_summary": (
                f"模式: {self.graph_mode}, "
                f"实体: {entity_counts}, "
                f"边: {edge_counts}"
            ),
        })

        return observation

    def is_done(self, observation: Dict) -> bool:
        total_edges = observation.get("total_edges", 0)
        has_saved = any(
            h["action"] == "save_edges" and h["result_status"] == "success"
            for h in self._action_history
        )

        if total_edges > 0 and has_saved:
            return True
        if len(self._action_history) >= 6:
            return True
        return False

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        action = action_plan.get("action", "")
        params = action_plan.get("params", {})

        action_map = {
            "collect_entities": self.action_collect_entities,
            "collect_posts": self.action_collect_posts,
            "build_author_edges": self.action_build_author_edges,
            "build_reply_edges": self.action_build_reply_edges,
            "build_belongsto_edges": self.action_build_belongsto_edges,
            "build_crosspost_edges": self.action_build_crosspost_edges,
            "build_cocomment_edges": self.action_build_cocomment_edges,
            "save_edges": self.action_save_edges,
            "validate_graph": self.action_validate_graph,
        }

        handler = action_map.get(action)
        if handler:
            return handler(params)
        elif action == "done":
            return {"status": "success", "message": "任务完成"}
        else:
            return {"status": "error", "message": f"未知行动: {action}"}

    # ==================== 数据获取 ====================

    def _get_posts(self) -> List[Dict]:
        """获取帖子列表"""
        if self.blackboard:
            posts = self.blackboard.read("subreddit_collector", "posts", [])
            if posts:
                return posts
        return self._current_task.get("posts", [])

    def _get_comments_map(self) -> Dict[str, List[Dict]]:
        """获取评论映射"""
        if self.blackboard:
            comments = self.blackboard.read("post_collector", "comments_map", {})
            if comments:
                return comments
        return self._current_task.get("comments_map", {})

    # ==================== 异构模式行动 ====================

    def action_collect_entities(self, params: Dict) -> Dict:
        """收集所有实体（异构模式）"""
        posts = self._get_posts()
        comments_map = self._get_comments_map()

        # 帖子
        for post in posts:
            self._entities["post"].add(post["post_id"])
            subreddit = post.get("subreddit", "")
            if subreddit:
                self._entities["subreddit"].add(subreddit)

            # 帖子作者
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                self._entities["user"].add(author)

        # 评论
        for post_id, comments in comments_map.items():
            for comment in comments:
                self._entities["comment"].add(comment["comment_id"])
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    self._entities["user"].add(author)

        entity_counts = {k: len(v) for k, v in self._entities.items()}
        logger.info(f"[SocialGraphBuilder] 实体收集(异构): {entity_counts}")

        return {
            "status": "success",
            "message": f"收集 {sum(entity_counts.values())} 个实体",
            "entity_counts": entity_counts,
        }

    def action_build_author_edges(self, params: Dict) -> Dict:
        """构建 author_of 边"""
        posts = self._get_posts()
        comments_map = self._get_comments_map()

        count = 0

        # user → post
        for post in posts:
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                self._edges["author_of"].append({
                    "source_id": f"user_{author}",
                    "source_type": "user",
                    "target_id": f"post_{post['post_id']}",
                    "target_type": "post",
                    "edge_type": "author_of",
                    "attributes": json.dumps({
                        "created_utc": post.get("created_utc", 0)
                    })
                })
                count += 1

        # user → comment
        for post_id, comments in comments_map.items():
            for comment in comments:
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    self._edges["author_of"].append({
                        "source_id": f"user_{author}",
                        "source_type": "user",
                        "target_id": f"comment_{comment['comment_id']}",
                        "target_type": "comment",
                        "edge_type": "author_of",
                        "attributes": json.dumps({
                            "created_utc": comment.get("created_utc", 0)
                        })
                    })
                    count += 1

        logger.info(f"[SocialGraphBuilder] author_of 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 author_of 边",
            "edge_count": count,
        }

    def action_build_reply_edges(self, params: Dict) -> Dict:
        """构建 reply_to 边"""
        comments_map = self._get_comments_map()

        count = 0
        for post_id, comments in comments_map.items():
            for comment in comments:
                parent_id = comment.get("parent_id", "")

                if parent_id.startswith("t3_"):
                    target = f"post_{parent_id[3:]}"
                    target_type = "post"
                elif parent_id.startswith("t1_"):
                    target = f"comment_{parent_id[3:]}"
                    target_type = "comment"
                else:
                    continue

                self._edges["reply_to"].append({
                    "source_id": f"comment_{comment['comment_id']}",
                    "source_type": "comment",
                    "target_id": target,
                    "target_type": target_type,
                    "edge_type": "reply_to",
                    "attributes": json.dumps({
                        "depth": comment.get("depth", 0)
                    })
                })
                count += 1

        logger.info(f"[SocialGraphBuilder] reply_to 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 reply_to 边",
            "edge_count": count,
        }

    def action_build_belongsto_edges(self, params: Dict) -> Dict:
        """构建 belongs_to 边"""
        posts = self._get_posts()

        count = 0
        for post in posts:
            subreddit = post.get("subreddit", "")
            if subreddit:
                self._edges["belongs_to"].append({
                    "source_id": f"post_{post['post_id']}",
                    "source_type": "post",
                    "target_id": f"subreddit_{subreddit}",
                    "target_type": "subreddit",
                    "edge_type": "belongs_to",
                    "attributes": "{}"
                })
                count += 1

        logger.info(f"[SocialGraphBuilder] belongs_to 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 belongs_to 边",
            "edge_count": count,
        }

    def action_build_crosspost_edges(self, params: Dict) -> Dict:
        """构建 crosspost 边"""
        posts = self._get_posts()
        post_ids = {p["post_id"] for p in posts}

        # 确保 defaultdict 键存在（即使无边也需要标记已构建）
        _ = self._edges["crosspost"]

        count = 0
        for post in posts:
            crosspost_parent = post.get("crosspost_parent", "")
            if crosspost_parent:
                parent_id = crosspost_parent.replace("t3_", "")
                if parent_id in post_ids:
                    self._edges["crosspost"].append({
                        "source_id": f"post_{post['post_id']}",
                        "source_type": "post",
                        "target_id": f"post_{parent_id}",
                        "target_type": "post",
                        "edge_type": "crosspost",
                        "attributes": json.dumps({
                            "original_subreddit": post.get("subreddit", "")
                        })
                    })
                    count += 1

        logger.info(f"[SocialGraphBuilder] crosspost 边: {count}")
        return {
            "status": "success",
            "message": f"构建 {count} 条 crosspost 边",
            "edge_count": count,
        }

    # ==================== 同构模式行动 (MAGB兼容) ====================

    def action_collect_posts(self, params: Dict) -> Dict:
        """
        收集帖子节点 + 构建用户→帖子映射（同构模式）

        算法：
        1. 遍历所有帖子，收集 post_id 集合
        2. 把帖子作者加入 user_to_posts 映射
        3. 遍历评论，把评论者也加入 user_to_posts 映射
        4. 构建标签映射：post_id → subreddit
        """
        posts = self._get_posts()
        comments_map = self._get_comments_map()

        self._user_to_posts = defaultdict(set)
        self._label_map = {}

        for post in posts:
            post_id = post["post_id"]
            self._entities["post"].add(post_id)

            # 帖子作者也算"参与"了该帖子
            author = post.get("author", "")
            if author and author not in ("[deleted]", "[removed]"):
                self._user_to_posts[author].add(post_id)

            # 标签 = subreddit
            subreddit = post.get("subreddit", "")
            if subreddit:
                self._label_map[post_id] = subreddit

        # 评论者
        for post_id, comments in comments_map.items():
            # 只处理在帖子集中的 post
            if post_id not in self._entities["post"]:
                continue
            for comment in comments:
                author = comment.get("author", "")
                if author and author not in ("[deleted]", "[removed]"):
                    self._user_to_posts[author].add(post_id)

        entity_counts = {k: len(v) for k, v in self._entities.items()}
        logger.info(
            f"[SocialGraphBuilder] 同构模式收集: "
            f"{len(self._entities['post'])} 帖子, "
            f"{len(self._user_to_posts)} 活跃用户, "
            f"{len(self._label_map)} 标签"
        )

        return {
            "status": "success",
            "message": f"收集 {len(self._entities['post'])} 个帖子节点",
            "entity_counts": entity_counts,
            "active_users": len(self._user_to_posts),
            "labeled_posts": len(self._label_map),
        }

    def action_build_cocomment_edges(self, params: Dict) -> Dict:
        """
        构建 co-commenting 边（同构模式, MAGB兼容）

        算法：
        1. 对每个参与过 >=2 个帖子的用户
        2. 生成所有帖子对的无向边
        3. 去重（同一帖子对可能被多个用户创建）
        """
        if not self._user_to_posts:
            # 如果还没收集，先自动调用
            self.action_collect_posts(params)

        edge_set = set()  # 用 frozenset 去重无向边
        count = 0

        for user, post_ids in self._user_to_posts.items():
            if len(post_ids) < 2:
                continue
            # 所有帖子对
            for p1, p2 in combinations(sorted(post_ids), 2):
                edge_key = frozenset([p1, p2])
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    # 无向边：存两条有向边 (i→j, j→i)
                    self._edges["co_comment"].append({
                        "source_id": f"post_{p1}",
                        "source_type": "post",
                        "target_id": f"post_{p2}",
                        "target_type": "post",
                        "edge_type": "co_comment",
                        "attributes": json.dumps({"user": user})
                    })
                    self._edges["co_comment"].append({
                        "source_id": f"post_{p2}",
                        "source_type": "post",
                        "target_id": f"post_{p1}",
                        "target_type": "post",
                        "edge_type": "co_comment",
                        "attributes": json.dumps({"user": user})
                    })
                    count += 2

        logger.info(
            f"[SocialGraphBuilder] co_comment 边: {count} "
            f"({count // 2} 无向边)"
        )
        return {
            "status": "success",
            "message": f"构建 {count} 条 co_comment 边 ({count // 2} 无向)",
            "edge_count": count,
        }

    # ==================== 共用行动 ====================

    def action_save_edges(self, params: Dict) -> Dict:
        """保存边到 CSV 文件"""
        task = self._current_task
        output_dir = task.get("output_dirs", {}).get("graphs", "./data/reddit_graphs")
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}
        total_edges = 0

        for edge_type, edges in self._edges.items():
            if not edges:
                continue

            file_path = os.path.join(output_dir, f"{edge_type}_edges.csv")
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["source_id", "source_type", "target_id",
                                "target_type", "edge_type", "attributes"]
                )
                writer.writeheader()
                writer.writerows(edges)

            saved_files[edge_type] = file_path
            total_edges += len(edges)

        # 写入 blackboard
        if self.blackboard:
            self.blackboard.write("social_graph", "edge_files", saved_files)
            self.blackboard.write("social_graph", "total_edges", total_edges)
            self.blackboard.write("social_graph", "edge_counts",
                                  {k: len(v) for k, v in self._edges.items()})
            self.blackboard.write("social_graph", "entity_counts",
                                  {k: len(v) for k, v in self._entities.items()})
            self.blackboard.write("social_graph", "graph_mode", self.graph_mode)
            if self.graph_mode == "homogeneous":
                self.blackboard.write("social_graph", "label_map", self._label_map)

        logger.info(
            f"[SocialGraphBuilder] 保存 {total_edges} 条边到 {len(saved_files)} 个文件"
        )

        return {
            "status": "success",
            "message": f"保存 {total_edges} 条边",
            "edge_files": saved_files,
            "total_edges": total_edges,
            "edge_counts": {k: len(v) for k, v in self._edges.items()},
        }

    def action_validate_graph(self, params: Dict) -> Dict:
        """验证图统计信息"""
        entity_counts = {k: len(v) for k, v in self._entities.items()}
        edge_counts = {k: len(v) for k, v in self._edges.items()}

        total_entities = sum(entity_counts.values())
        total_edges = sum(edge_counts.values())

        stats = {
            "graph_mode": self.graph_mode,
            "entity_counts": entity_counts,
            "edge_counts": edge_counts,
            "total_entities": total_entities,
            "total_edges": total_edges,
            "avg_degree": total_edges / max(total_entities, 1),
        }

        if self.graph_mode == "homogeneous":
            stats["label_distribution"] = {
                label: sum(1 for l in self._label_map.values() if l == label)
                for label in set(self._label_map.values())
            }

        issues = []
        if total_edges < 10:
            issues.append("边数过少")

        return {
            "status": "success",
            "message": f"图验证: {stats}",
            "stats": stats,
            "issues": issues,
        }

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        """执行入口"""
        self._current_task = task
        self._accumulated_results = {}
        self._edges = defaultdict(list)
        self._entities = defaultdict(set)
        self._user_to_posts = defaultdict(set)
        self._label_map = {}

        # 从 task dict 读取 graph_mode
        self.graph_mode = task.get("graph_mode", "heterogeneous")
        logger.info(
            f"[SocialGraphBuilder] 启动, graph_mode={self.graph_mode}"
        )

        return self.run_react_loop(
            task=task,
            blackboard=self.blackboard or Blackboard(),
            message_bus=self.message_bus or MessageBus()
        )

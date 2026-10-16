#!/usr/bin/env python3
"""
直接从 Pushshift 预处理数据构建 Reddit 多模态属性图数据集
绕过 pipeline 的 LLM rate limit 问题

输入:
  - data/reddit_raw/posts.json (Pushshift 过滤后的帖子)
  - data/reddit_raw/comments.json (评论，可能为空)
  - data/reddit_raw/users.json (用户信息)
  - data/reddit_raw/subreddits.json (subreddit 信息)

输出:
  - data/reddit_datasets/reddit_multimodal_graph.json
"""
import os
import json
import csv
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from loguru import logger


DATA_DIR = "./data/reddit_raw"
OUTPUT_DIR = "./data/reddit_datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_multimodal_graph.json")


def load_json(path):
    if not os.path.exists(path) or os.path.getsize(path) <= 2:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main():
    logger.info("=" * 60)
    logger.info("Reddit 多模态属性图数据集构建")
    logger.info("=" * 60)

    # ==================== 加载数据 ====================
    posts = load_json(os.path.join(DATA_DIR, "posts.json"))
    if isinstance(posts, dict):
        posts = list(posts.values())
    if not isinstance(posts, list):
        posts = []

    comments_map = load_json(os.path.join(DATA_DIR, "comments.json"))
    if not isinstance(comments_map, dict):
        comments_map = {}

    users_data = load_json(os.path.join(DATA_DIR, "users.json"))
    if isinstance(users_data, list):
        users_data = {u.get("username", ""): u for u in users_data if isinstance(u, dict)}

    subreddits_data = load_json(os.path.join(DATA_DIR, "subreddits.json"))
    if isinstance(subreddits_data, list):
        subreddits_data = {s.get("name", ""): s for s in subreddits_data if isinstance(s, dict)}

    logger.info(f"加载: {len(posts)} 帖子, {sum(len(v) for v in comments_map.values())} 评论, "
                f"{len(users_data)} 用户, {len(subreddits_data)} subreddit")

    # ==================== 构建节点 ====================
    nodes = []
    seen_ids = set()
    node_type_counts = defaultdict(int)

    # --- Subreddit 节点 ---
    subreddit_names = set()
    for post in posts:
        sub_name = post.get("subreddit", "")
        if sub_name:
            subreddit_names.add(sub_name)

    for name in sorted(subreddit_names):
        node_id = f"subreddit_{name}"
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            sub_info = subreddits_data.get(name, {})
            nodes.append({
                "id": node_id,
                "type": "subreddit",
                "attributes": {
                    "name": name,
                    "subscribers": sub_info.get("subscribers", 0),
                    "description_length": sub_info.get("description_length", 0),
                }
            })
            node_type_counts["subreddit"] += 1

    # --- User 节点 ---
    user_post_count = defaultdict(int)
    user_comment_count = defaultdict(int)
    post_authors = {}

    for post in posts:
        author = post.get("author", "")
        if author and author != "[deleted]":
            user_post_count[author] += 1
            post_authors[post["post_id"]] = author

    for post_id, comments in comments_map.items():
        for comment in comments:
            author = comment.get("author", "")
            if author and author != "[deleted]":
                user_comment_count[author] += 1

    all_users = set(user_post_count.keys()) | set(user_comment_count.keys())
    for username in sorted(all_users):
        node_id = f"user_{username}"
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            user_info = users_data.get(username, {})
            nodes.append({
                "id": node_id,
                "type": "user",
                "attributes": {
                    "username": username,
                    "post_count": user_post_count.get(username, 0),
                    "comment_count": user_comment_count.get(username, 0),
                    "total_karma": user_info.get("total_karma", 0),
                    "link_karma": user_info.get("link_karma", 0),
                    "comment_karma": user_info.get("comment_karma", 0),
                }
            })
            node_type_counts["user"] += 1

    # --- Post 节点 (多模态属性：text + image + numeric) ---
    for post in posts:
        node_id = f"post_{post['post_id']}"
        if node_id not in seen_ids:
            seen_ids.add(node_id)

            # 多模态属性
            attrs = {
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "title_length": post.get("title_length", len(post.get("title", ""))),
                "text_length": post.get("text_length", 0),
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio", 0),
                "num_comments": post.get("num_comments", 0),
                "has_image": post.get("has_image", False),
                "image_count": post.get("image_count", 0),
                "created_utc": post.get("created_utc", 0),
                "link_flair_text": post.get("link_flair_text", ""),
                "subreddit": post.get("subreddit", ""),
                "is_self": post.get("is_self", False),
                "over_18": post.get("over_18", False),
            }
            # 图片URL（多模态关键）
            if post.get("url"):
                attrs["url"] = post["url"]
            if post.get("thumbnail") and isinstance(post["thumbnail"], str) and post["thumbnail"].startswith("http"):
                attrs["thumbnail"] = post["thumbnail"]

            nodes.append({"id": node_id, "type": "post", "attributes": attrs})
            node_type_counts["post"] += 1

    # --- Comment 节点 ---
    comment_count = 0
    for post_id, comments in comments_map.items():
        for comment in comments:
            node_id = f"comment_{comment['comment_id']}"
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "type": "comment",
                    "attributes": {
                        "body": comment.get("body", ""),
                        "text_length": comment.get("body_length", 0),
                        "score": comment.get("score", 0),
                        "is_submitter": comment.get("is_submitter", False),
                        "created_utc": comment.get("created_utc", 0),
                        "depth": comment.get("depth", 0),
                    }
                })
                node_type_counts["comment"] += 1
                comment_count += 1

    logger.info(f"节点: {len(nodes)} ({dict(node_type_counts)})")

    # ==================== 构建边 ====================
    edges = []
    edge_type_counts = defaultdict(int)

    # --- belongs_to: post → subreddit ---
    for post in posts:
        sub_name = post.get("subreddit", "")
        if sub_name:
            edges.append({
                "source": f"post_{post['post_id']}",
                "target": f"subreddit_{sub_name}",
                "type": "belongs_to",
                "attributes": {}
            })
            edge_type_counts["belongs_to"] += 1

    # --- author_of: user → post ---
    for post in posts:
        author = post.get("author", "")
        if author and author != "[deleted]":
            edges.append({
                "source": f"user_{author}",
                "target": f"post_{post['post_id']}",
                "type": "author_of",
                "attributes": {}
            })
            edge_type_counts["author_of"] += 1

    # --- reply_to: comment → post / comment → comment ---
    for post_id, comments in comments_map.items():
        for comment in comments:
            parent_id = comment.get("parent_id", "")
            if parent_id.startswith("t3_"):
                # reply to post
                target_id = parent_id[3:]
                edges.append({
                    "source": f"comment_{comment['comment_id']}",
                    "target": f"post_{target_id}",
                    "type": "reply_to",
                    "attributes": {"depth": comment.get("depth", 0)}
                })
                edge_type_counts["reply_to"] += 1
            elif parent_id.startswith("t1_"):
                # reply to comment
                target_id = parent_id[3:]
                edges.append({
                    "source": f"comment_{comment['comment_id']}",
                    "target": f"comment_{target_id}",
                    "type": "reply_to",
                    "attributes": {"depth": comment.get("depth", 0)}
                })
                edge_type_counts["reply_to"] += 1

    # --- crosspost: post → post ---
    for post in posts:
        crosspost_parent = post.get("crosspost_parent", "")
        if crosspost_parent:
            edges.append({
                "source": f"post_{post['post_id']}",
                "target": f"post_{crosspost_parent}",
                "type": "crosspost",
                "attributes": {}
            })
            edge_type_counts["crosspost"] += 1

    # --- co_comment: post ↔ post (同构边，同一用户在两个帖子下都评论过) ---
    # 构建 user → posts 映射
    user_to_posts = defaultdict(set)
    for post in posts:
        author = post.get("author", "")
        if author and author != "[deleted]":
            user_to_posts[author].add(post["post_id"])

    # 也从评论中收集（如果有评论数据）
    for post_id, comments in comments_map.items():
        for comment in comments:
            author = comment.get("author", "")
            if author and author != "[deleted]":
                user_to_posts[author].add(post_id)

    # 为每个在 >=2 个帖子中活跃的用户，生成帖子对
    edge_set = set()
    for user, post_ids in user_to_posts.items():
        if len(post_ids) >= 2:
            sorted_ids = sorted(post_ids)
            for p1, p2 in combinations(sorted_ids, 2):
                edge_key = (p1, p2)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    # 无向边：两条有向边
                    edges.append({
                        "source": f"post_{p1}",
                        "target": f"post_{p2}",
                        "type": "co_comment",
                        "attributes": {}
                    })
                    edges.append({
                        "source": f"post_{p2}",
                        "target": f"post_{p1}",
                        "type": "co_comment",
                        "attributes": {}
                    })
                    edge_type_counts["co_comment"] += 2

    logger.info(f"边: {len(edges)} ({dict(edge_type_counts)})")

    # ==================== 构建元数据 ====================
    total_images = sum(n["attributes"].get("image_count", 0) for n in nodes if n["type"] == "post")
    has_image_posts = sum(1 for n in nodes if n["type"] == "post" and n["attributes"].get("has_image"))
    total_text_chars = sum(
        n["attributes"].get("text_length", 0) + n["attributes"].get("title_length", 0)
        for n in nodes if n["type"] == "post"
    )

    # 标签分布
    label_dist = defaultdict(int)
    for n in nodes:
        if n["type"] == "post":
            label_dist[n["attributes"].get("subreddit", "unknown")] += 1

    metadata = {
        "name": "reddit_multimodal_graph",
        "domain": "reddit",
        "graph_mode": "heterogeneous",
        "data_source": "pushshift",
        "created_at": datetime.now().isoformat(),
        "statistics": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_distribution": dict(node_type_counts),
            "edge_type_distribution": dict(edge_type_counts),
            "total_images": total_images,
            "posts_with_images": has_image_posts,
            "total_text_chars": total_text_chars,
            "label_distribution": dict(label_dist),
            "unique_users": len(all_users),
            "unique_subreddits": len(subreddit_names),
        },
        "schema": {
            "node_types": list(node_type_counts.keys()),
            "edge_types": list(edge_type_counts.keys()),
            "node_attributes": {
                "post": ["title", "selftext", "url", "score", "has_image", "image_count",
                         "subreddit", "link_flair_text", "upvote_ratio", "created_utc"],
                "user": ["username", "post_count", "comment_count", "total_karma"],
                "subreddit": ["name", "subscribers"],
                "comment": ["body", "text_length", "score", "depth"],
            },
            "modalities": ["text", "image", "numeric"],
        }
    }

    # ==================== 组装并保存 ====================
    dataset = {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 数据集已保存: {OUTPUT_FILE}")
    logger.info(f"   节点: {len(nodes)} ({dict(node_type_counts)})")
    logger.info(f"   边: {len(edges)} ({dict(edge_type_counts)})")
    logger.info(f"   多模态: 文本({total_text_chars} chars) + 图片({total_images}, {has_image_posts} posts)")
    logger.info(f"   标签分布: {dict(label_dist)}")

    return dataset


if __name__ == "__main__":
    main()

"""
Pushshift Reddit 数据加载器
从 Pushshift 格式的 .zst 文件中流式加载 Reddit submissions 和 comments，
按目标 subreddit 过滤，输出与 RedditToolManager 兼容的格式。

使用方法:
    loader = PushshiftDataLoader(data_dir="./data/reddit_raw")
    loader.download_and_filter(
        subreddits=["MachineLearning", "datascience", ...],
        posts_per_subreddit=2000,
        comments_per_post=25
    )
    posts = loader.get_posts()
    comments = loader.get_comments()
"""
import os
import json
import gzip
import time
import hashlib
import random
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from loguru import logger


class PushshiftDataLoader:
    """从 Pushshift 数据文件加载 Reddit 数据"""

    # Zenodo 上的 Pushshift 数据集 (2019年4月)
    ZENODO_BASE = "https://zenodo.org/api/records/3608135/files"

    # 目标 subreddit 及其分类标签（用于 node classification）
    DEFAULT_SUBREDDITS = [
        "MachineLearning",
        "datascience",
        "artificial",
        "deeplearning",
        "learnmachinelearning",
        "statistics",
        "computerscience",
        "artificial",
        "technology",
        "science",
    ]

    def __init__(self, data_dir: str = "./data/reddit_raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self._posts: List[Dict] = []
        self._comments: Dict[str, List[Dict]] = {}  # post_id -> [comments]
        self._subreddit_stats: Dict[str, int] = defaultdict(int)

    def load_from_file(self, submissions_path: str = None,
                       comments_path: str = None,
                       subreddits: List[str] = None,
                       posts_per_subreddit: int = 2000,
                       comments_per_post: int = 25) -> Dict[str, Any]:
        """
        从本地 Pushshift 文件加载 Reddit 数据

        支持的文件格式:
        - .zst (Zstandard 压缩)
        - .gz (gzip 压缩)
        - .json (纯 JSON Lines)
        - .jsonl (JSON Lines)

        Args:
            submissions_path: submissions 文件路径
            comments_path: comments 文件路径
            subreddits: 目标 subreddit 列表
            posts_per_subreddit: 每个 subreddit 最多保留的帖子数
            comments_per_post: 每个帖子最多保留的评论数

        Returns:
            加载统计信息
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        target_subs = set(s.lower() for s in subreddits)

        # 1. 加载 submissions
        if submissions_path and os.path.exists(submissions_path):
            logger.info(f"[PushshiftLoader] 加载 submissions: {submissions_path}")
            self._load_submissions(
                submissions_path, target_subs, posts_per_subreddit
            )

        # 2. 加载 comments
        if comments_path and os.path.exists(comments_path):
            post_ids = set(p["post_id"] for p in self._posts)
            logger.info(
                f"[PushshiftLoader] 加载 comments: {comments_path}, "
                f"目标帖子数: {len(post_ids)}"
            )
            self._load_comments(
                comments_path, post_ids, comments_per_post
            )

        # 3. 保存过滤后的数据
        self._save_filtered_data()

        stats = {
            "total_posts": len(self._posts),
            "total_comments": sum(len(c) for c in self._comments.values()),
            "subreddit_distribution": dict(self._subreddit_stats),
            "unique_authors": len(set(
                p["author"] for p in self._posts if p["author"] != "[deleted]"
            )),
        }

        logger.info(f"[PushshiftLoader] 加载完成: {stats}")
        return stats

    def _open_file(self, path: str):
        """根据文件扩展名打开文件，返回行迭代器"""
        if path.endswith('.zst'):
            return self._open_zst(path)
        elif path.endswith('.gz'):
            return gzip.open(path, 'rt', encoding='utf-8', errors='replace')
        else:
            return open(path, 'r', encoding='utf-8', errors='replace')

    def _open_zst(self, path: str):
        """打开 .zst 文件"""
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            fh = open(path, 'rb')
            stream = dctx.stream_reader(fh)
            import io
            return io.TextIOWrapper(stream, encoding='utf-8', errors='replace')
        except ImportError:
            # fallback: 使用系统 zstd 命令
            import subprocess
            proc = subprocess.Popen(
                ['zstd', '-d', path, '-c', '--stream'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            import io
            return io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')

    def _load_submissions(self, path: str, target_subs: Set[str],
                          posts_per_subreddit: int):
        """加载 submissions 并按 subreddit 过滤"""
        subreddit_posts = defaultdict(list)
        total_scanned = 0

        fh = self._open_file(path)
        try:
            for line in fh:
                total_scanned += 1
                if total_scanned % 500000 == 0:
                    logger.info(
                        f"[PushshiftLoader] 扫描 submissions: "
                        f"{total_scanned}, 收集: {sum(len(v) for v in subreddit_posts.values())}"
                    )

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                sub = item.get("subreddit", "").lower()
                if sub not in target_subs:
                    continue

                # 已经收集够了这个 subreddit
                if len(subreddit_posts[sub]) >= posts_per_subreddit:
                    continue

                post = self._parse_pushshift_submission(item)
                if post:
                    subreddit_posts[sub].append(post)

            # 检查是否所有 subreddit 都收集够了
            all_full = all(
                len(subreddit_posts[s]) >= posts_per_subreddit
                for s in target_subs
            )
            if all_full:
                logger.info("[PushshiftLoader] 所有 subreddit 已收集够")

        finally:
            fh.close()

        # 合并所有帖子
        self._posts = []
        for sub in target_subs:
            posts = subreddit_posts.get(sub, [])
            self._posts.extend(posts)
            self._subreddit_stats[sub] = len(posts)

        logger.info(
            f"[PushshiftLoader] Submissions: 扫描 {total_scanned}, "
            f"收集 {len(self._posts)} 帖子 "
            f"(来自 {len(self._subreddit_stats)} 个 subreddit)"
        )

    def _load_comments(self, path: str, target_post_ids: Set[str],
                       comments_per_post: int):
        """加载 comments 并按 post_id 过滤"""
        comments_map = defaultdict(list)
        total_scanned = 0
        matched = 0

        fh = self._open_file(path)
        try:
            for line in fh:
                total_scanned += 1
                if total_scanned % 1000000 == 0:
                    logger.info(
                        f"[PushshiftLoader] 扫描 comments: "
                        f"{total_scanned}, 匹配: {matched}"
                    )

                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Pushshift 使用 link_id (t3_xxx) 表示帖子
                link_id = item.get("link_id", "")
                post_id = link_id.replace("t3_", "")

                if post_id not in target_post_ids:
                    continue

                # 限制每帖评论数
                if len(comments_map[post_id]) >= comments_per_post:
                    continue

                comment = self._parse_pushshift_comment(item, post_id)
                if comment:
                    comments_map[post_id].append(comment)
                    matched += 1

        finally:
            fh.close()

        self._comments = dict(comments_map)
        logger.info(
            f"[PushshiftLoader] Comments: 扫描 {total_scanned}, "
            f"匹配 {matched} 评论 ({len(self._comments)} 个帖子)"
        )

    def _parse_pushshift_submission(self, item: Dict) -> Optional[Dict]:
        """将 Pushshift submission 转换为 RedditToolManager 兼容格式"""
        try:
            post_id = item.get("id", "")
            if not post_id:
                return None

            selftext = item.get("selftext", "") or ""
            title = item.get("title", "") or ""
            url = item.get("url", "") or ""
            author = item.get("author", "") or "[deleted]"

            # 判断是否有图片
            has_image = False
            if url:
                image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
                url_lower = url.lower()
                if any(url_lower.endswith(ext) for ext in image_exts):
                    has_image = True
                if 'i.redd.it' in url_lower or 'i.imgur.com' in url_lower:
                    has_image = True
                if 'reddit.com/gallery' in url_lower:
                    has_image = True

            # 处理 crosspost
            crosspost_parent = None
            if item.get("crosspost_parent"):
                crosspost_parent = item["crosspost_parent"]

            # 处理 flair
            flair = item.get("link_flair_text", "") or ""

            # 处理 thumbnail
            thumbnail = item.get("thumbnail", "") or ""
            if thumbnail and thumbnail.startswith("http"):
                has_image = True

            return {
                "post_id": post_id,
                "title": title,
                "selftext": selftext,
                "text_length": len(selftext),
                "title_length": len(title),
                "author": author,
                "subreddit": item.get("subreddit", ""),
                "score": item.get("score", 0),
                "upvote_ratio": item.get("upvote_ratio", 0.0),
                "num_comments": item.get("num_comments", 0),
                "created_utc": int(item.get("created_utc", 0)),
                "url": url,
                "is_self": item.get("is_self", False),
                "has_image": has_image,
                "image_count": 1 if has_image else 0,
                "link_flair_text": flair,
                "crosspost_parent": crosspost_parent,
                "over_18": item.get("over_18", False),
                "spoiler": item.get("spoiler", False),
                "permalink": item.get("permalink", f"/r/{item.get('subreddit', '')}/comments/{post_id}/"),
                "thumbnail": thumbnail,
            }
        except Exception as e:
            logger.debug(f"[PushshiftLoader] 解析 submission 失败: {e}")
            return None

    def _parse_pushshift_comment(self, item: Dict, post_id: str) -> Optional[Dict]:
        """将 Pushshift comment 转换为 RedditToolManager 兼容格式"""
        try:
            comment_id = item.get("id", "")
            if not comment_id:
                return None

            body = item.get("body", "") or ""
            author = item.get("author", "") or "[deleted]"

            return {
                "comment_id": comment_id,
                "post_id": post_id,
                "parent_id": item.get("parent_id", f"t3_{post_id}"),
                "author": author,
                "body": body,
                "body_length": len(body),
                "score": item.get("score", 0),
                "created_utc": int(item.get("created_utc", 0)),
                "depth": 0,  # 将在后面计算
                "is_submitter": item.get("is_submitter", False),
                "controversiality": item.get("controversiality", 0),
            }
        except Exception as e:
            logger.debug(f"[PushshiftLoader] 解析 comment 失败: {e}")
            return None

    def _save_filtered_data(self):
        """保存过滤后的数据为 JSON"""
        # 保存帖子
        posts_path = os.path.join(self.data_dir, "posts.json")
        with open(posts_path, 'w', encoding='utf-8') as f:
            json.dump(self._posts, f, ensure_ascii=False, indent=2)
        logger.info(f"[PushshiftLoader] 保存 {len(self._posts)} 帖子到 {posts_path}")

        # 保存评论
        comments_path = os.path.join(self.data_dir, "comments.json")
        with open(comments_path, 'w', encoding='utf-8') as f:
            json.dump(self._comments, f, ensure_ascii=False, indent=2)
        total_comments = sum(len(c) for c in self._comments.values())
        logger.info(f"[PushshiftLoader] 保存 {total_comments} 评论到 {comments_path}")

        # 保存 subreddit 信息
        subreddits_info = {}
        for post in self._posts:
            sub = post["subreddit"]
            if sub not in subreddits_info:
                subreddits_info[sub] = {
                    "name": sub,
                    "post_count": 0,
                }
            subreddits_info[sub]["post_count"] += 1

        subs_path = os.path.join(self.data_dir, "subreddits.json")
        with open(subs_path, 'w', encoding='utf-8') as f:
            json.dump(subreddits_info, f, ensure_ascii=False, indent=2)

        # 计算评论深度
        self._compute_comment_depths()

    def _compute_comment_depths(self):
        """根据 parent_id 计算评论深度"""
        for post_id, comments in self._comments.items():
            # 建立评论 ID -> 评论 映射
            comment_map = {c["comment_id"]: c for c in comments}
            for comment in comments:
                parent = comment["parent_id"]
                # t3_ 开头的是帖子，深度为 0
                if parent.startswith("t3_"):
                    comment["depth"] = 0
                else:
                    # t1_ 开头的是回复评论
                    parent_comment_id = parent.replace("t1_", "")
                    if parent_comment_id in comment_map:
                        comment["depth"] = min(
                            comment_map[parent_comment_id].get("depth", 0) + 1, 5
                        )
                    else:
                        comment["depth"] = 1  # 未知父评论，设为 1

    def get_posts(self) -> List[Dict]:
        """获取所有帖子"""
        return self._posts

    def get_comments(self) -> Dict[str, List[Dict]]:
        """获取评论映射"""
        return self._comments

    def get_users(self) -> List[Dict]:
        """从帖子和评论中提取唯一用户信息"""
        users = {}
        # 从帖子中提取用户
        for post in self._posts:
            author = post["author"]
            if author and author != "[deleted]":
                if author not in users:
                    users[author] = {
                        "username": author,
                        "post_count": 0,
                        "comment_count": 0,
                    }
                users[author]["post_count"] += 1

        # 从评论中提取用户
        for post_id, comments in self._comments.items():
            for comment in comments:
                author = comment["author"]
                if author and author != "[deleted]":
                    if author not in users:
                        users[author] = {
                            "username": author,
                            "post_count": 0,
                            "comment_count": 0,
                        }
                    users[author]["comment_count"] += 1

        return list(users.values())

    def load_precomputed(self) -> bool:
        """
        加载之前计算并保存的过滤数据

        Returns:
            True 如果成功加载了预计算数据
        """
        posts_path = os.path.join(self.data_dir, "posts.json")
        comments_path = os.path.join(self.data_dir, "comments.json")

        if not os.path.exists(posts_path):
            return False

        try:
            with open(posts_path, 'r', encoding='utf-8') as f:
                self._posts = json.load(f)

            if os.path.exists(comments_path):
                with open(comments_path, 'r', encoding='utf-8') as f:
                    self._comments = json.load(f)

            # 重建 subreddit 统计
            for post in self._posts:
                sub = post.get("subreddit", "unknown").lower()
                self._subreddit_stats[sub] += 1

            logger.info(
                f"[PushshiftLoader] 加载预计算数据: "
                f"{len(self._posts)} 帖子, "
                f"{sum(len(c) for c in self._comments.values())} 评论"
            )
            return True
        except Exception as e:
            logger.error(f"[PushshiftLoader] 加载预计算数据失败: {e}")
            return False

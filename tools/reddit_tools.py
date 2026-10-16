"""
Reddit 数据采集工具集
封装 PRAW (Python Reddit API Wrapper) 提供统一的数据采集接口

功能:
- Subreddit 发帖采集（hot/new/top/rising）
- 评论树采集（深度限制 BFS）
- 帖子图片下载
- 用户元数据采集
- 速率限制管理
"""
import os
import re
import json
import time
import hashlib
import urllib.request
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


class RedditToolManager:
    """Reddit 数据采集工具管理器"""

    def __init__(self, client_id: str = None, client_secret: str = None,
                 user_agent: str = "DataCrawlerAgent/1.0 (Research)"):
        """
        初始化 Reddit 工具

        Args:
            client_id: Reddit App client_id
            client_secret: Reddit App client_secret
            user_agent: User-Agent 字符串
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = user_agent
        self._reddit = None
        self._request_count = 0
        self._last_request_time = 0
        self._min_interval = 1.0  # 秒，最小请求间隔

    def _get_reddit(self):
        """延迟初始化 PRAW Reddit 实例"""
        if self._reddit is None:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("[RedditTools] PRAW 初始化成功")
            except ImportError:
                raise ImportError("请安装 praw: pip install praw")
            except Exception as e:
                logger.error(f"[RedditTools] PRAW 初始化失败: {e}")
                raise
        return self._reddit

    def _rate_limit(self):
        """简单的速率限制"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    # ==================== Subreddit 采集 ====================

    def fetch_subreddit_posts(self, subreddit_name: str, sort: str = "hot",
                              limit: int = 100, time_filter: str = "year") -> List[Dict]:
        """
        从指定 subreddit 采集帖子

        Args:
            subreddit_name: subreddit 名称（不含 r/）
            sort: 排序方式 (hot/new/top/rising/controversial)
            limit: 最大帖子数
            time_filter: 时间过滤 (hour/day/week/month/year/all)

        Returns:
            帖子元数据列表
        """
        reddit = self._get_reddit()
        self._rate_limit()

        try:
            subreddit = reddit.subreddit(subreddit_name)

            if sort == "hot":
                submissions = subreddit.hot(limit=limit)
            elif sort == "new":
                submissions = subreddit.new(limit=limit)
            elif sort == "top":
                submissions = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == "rising":
                submissions = subreddit.rising(limit=limit)
            elif sort == "controversial":
                submissions = subreddit.controversial(time_filter=time_filter, limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)

            posts = []
            for submission in submissions:
                post_data = self._parse_submission(submission)
                if post_data:
                    posts.append(post_data)

            logger.info(
                f"[RedditTools] 采集 r/{subreddit_name} ({sort}): "
                f"{len(posts)} 帖子"
            )
            return posts

        except Exception as e:
            logger.error(f"[RedditTools] 采集 r/{subreddit_name} 失败: {e}")
            return []

    def fetch_subreddit_info(self, subreddit_name: str) -> Optional[Dict]:
        """获取 subreddit 元信息"""
        reddit = self._get_reddit()
        self._rate_limit()

        try:
            sub = reddit.subreddit(subreddit_name)
            return {
                "name": sub.display_name,
                "subscribers": sub.subscribers,
                "description": sub.public_description or "",
                "description_length": len(sub.public_description or ""),
                "created_utc": int(sub.created_utc),
                "over18": sub.over18,
                "active_user_count": sub.active_user_count,
            }
        except Exception as e:
            logger.error(f"[RedditTools] 获取 r/{subreddit_name} 信息失败: {e}")
            return None

    def search_subreddits(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索相关 subreddit"""
        reddit = self._get_reddit()
        self._rate_limit()

        try:
            results = reddit.subreddits.search(query, limit=limit)
            subreddits = []
            for sub in results:
                subreddits.append({
                    "name": sub.display_name,
                    "subscribers": sub.subscribers,
                    "public_description": (sub.public_description or "")[:200],
                })
            return subreddits
        except Exception as e:
            logger.error(f"[RedditTools] 搜索 subreddit 失败: {e}")
            return []

    # ==================== 评论采集 ====================

    def fetch_post_comments(self, post_id: str, max_depth: int = 5,
                            limit: int = 25) -> List[Dict]:
        """
        采集帖子评论树（深度限制 BFS）

        Args:
            post_id: 帖子 ID
            max_depth: 最大评论深度
            limit: 每层最大评论数

        Returns:
            评论列表（扁平化，含 depth 和 parent_id）
        """
        reddit = self._get_reddit()
        self._rate_limit()

        try:
            submission = reddit.submission(id=post_id)
            submission.comment_sort = "best"
            submission.comments.replace_more(limit=0)  # 移除 "load more" 占位符

            comments = []
            self._extract_comments_recursive(
                submission.comments, comments,
                parent_id=f"t3_{post_id}",  # t3 = submission type prefix
                depth=0, max_depth=max_depth, limit=limit,
                post_id=post_id
            )

            logger.info(
                f"[RedditTools] 采集帖子 {post_id} 评论: "
                f"{len(comments)} 条"
            )
            return comments

        except Exception as e:
            logger.error(f"[RedditTools] 采集帖子 {post_id} 评论失败: {e}")
            return []

    def _extract_comments_recursive(self, comment_forest, results: List[Dict],
                                     parent_id: str, depth: int,
                                     max_depth: int, limit: int,
                                     post_id: str):
        """递归提取评论树"""
        if depth > max_depth:
            return

        count = 0
        for comment in comment_forest:
            if count >= limit:
                break

            # 跳过 MoreComments 占位符
            if hasattr(comment, 'body'):  # 真正的 Comment 对象
                comment_data = {
                    "comment_id": comment.id,
                    "post_id": post_id,
                    "parent_id": parent_id,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "body": comment.body or "",
                    "body_length": len(comment.body) if comment.body else 0,
                    "score": comment.score,
                    "created_utc": int(comment.created_utc),
                    "depth": depth,
                    "is_submitter": comment.is_submitter,
                    "controversiality": comment.controversiality,
                }
                results.append(comment_data)
                count += 1

                # 递归子评论
                if hasattr(comment, 'replies') and comment.replies:
                    self._extract_comments_recursive(
                        comment.replies, results,
                        parent_id=f"t1_{comment.id}",  # t1 = comment type prefix
                        depth=depth + 1, max_depth=max_depth,
                        limit=limit, post_id=post_id
                    )

    def fetch_batch_post_comments(self, post_ids: List[str],
                                   max_depth: int = 3, limit: int = 20,
                                   max_workers: int = 3) -> Dict[str, List[Dict]]:
        """批量采集多个帖子的评论"""
        results = {}

        def fetch_one(post_id):
            comments = self.fetch_post_comments(
                post_id, max_depth=max_depth, limit=limit
            )
            return post_id, comments

        # 限制并发数，避免 API 限流
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, pid): pid for pid in post_ids}
            for future in as_completed(futures):
                pid, comments = future.result()
                results[pid] = comments
                logger.info(
                    f"[RedditTools] 批量评论采集进度: "
                    f"{len(results)}/{len(post_ids)}"
                )

        total = sum(len(c) for c in results.values())
        logger.info(
            f"[RedditTools] 批量评论采集完成: "
            f"{len(results)} 帖子, {total} 评论"
        )
        return results

    # ==================== 图片下载 ====================

    def download_reddit_images(self, posts: List[Dict], output_dir: str,
                               max_workers: int = 5) -> Dict[str, Any]:
        """
        从帖子列表下载图片

        Args:
            posts: 帖子列表（需含 url 和 post_id 字段）
            output_dir: 图片保存目录
            max_workers: 并发下载数

        Returns:
            下载结果统计
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {"downloaded": 0, "skipped": 0, "failed": 0, "details": {}}

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        def download_one(post: Dict) -> tuple:
            post_id = post.get("post_id", post.get("id", ""))
            image_url = post.get("url", "")

            if not image_url or not post_id:
                return post_id, {"status": "skipped", "reason": "missing url or id"}

            # 检查 URL 是否指向图片
            url_lower = image_url.lower()
            is_image = any(url_lower.endswith(ext) for ext in image_extensions)

            # 处理 imgur 等图片托管（无扩展名的图片 URL）
            if not is_image and ('imgur.com' in url_lower or 'i.redd.it' in url_lower):
                if not url_lower.endswith(('.jpg', '.png', '.gif')):
                    image_url += '.jpg'  # 尝试添加扩展名
                is_image = True

            if not is_image:
                return post_id, {"status": "skipped", "reason": "not an image URL"}

            post_img_dir = os.path.join(output_dir, post_id)
            os.makedirs(post_img_dir, exist_ok=True)

            # 生成文件名
            ext = '.jpg'
            for e in image_extensions:
                if image_url.lower().endswith(e):
                    ext = e
                    break

            save_path = os.path.join(post_img_dir, f"post_image{ext}")

            if os.path.exists(save_path):
                return post_id, {
                    "status": "cached",
                    "path": save_path,
                    "image_count": 1
                }

            try:
                # 直连，不使用代理
                req = urllib.request.Request(
                    image_url,
                    headers={'User-Agent': self.user_agent}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(save_path, 'wb') as f:
                        f.write(response.read())

                return post_id, {
                    "status": "downloaded",
                    "path": save_path,
                    "image_count": 1
                }
            except Exception as e:
                return post_id, {
                    "status": "failed",
                    "reason": str(e)[:200]
                }

        # 只下载有图片的帖子
        image_posts = [p for p in posts if p.get("has_image", False) or
                       self._is_image_post(p.get("url", ""))]

        logger.info(
            f"[RedditTools] 开始下载 {len(image_posts)} 张图片 "
            f"(共 {len(posts)} 帖子)"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, p): p for p in image_posts}
            for future in as_completed(futures):
                post_id, detail = future.result()
                results["details"][post_id] = detail

                status = detail["status"]
                if status == "downloaded":
                    results["downloaded"] += 1
                elif status == "cached":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1

        logger.info(
            f"[RedditTools] 图片下载完成: "
            f"下载 {results['downloaded']}, "
            f"缓存 {results['skipped']}, "
            f"失败 {results['failed']}"
        )
        return results

    def _is_image_post(self, url: str) -> bool:
        """判断 URL 是否为图片"""
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in image_exts):
            return True
        if 'i.redd.it' in url_lower or 'i.imgur.com' in url_lower:
            return True
        # reddit gallery
        if 'reddit.com/gallery' in url_lower:
            return True
        return False

    # ==================== 帖子解析 ====================

    def _parse_submission(self, submission) -> Optional[Dict]:
        """
        解析 Reddit Submission 对象为结构化字典

        Args:
            submission: praw.models.Submission

        Returns:
            结构化帖子数据
        """
        try:
            # 判断是否有图片
            url = submission.url or ""
            has_image = self._is_image_post(url)
            is_self = submission.is_self

            # 处理 self post 的文本
            selftext = submission.selftext or ""
            title = submission.title or ""

            # 处理被删除的帖子
            if submission.author is None:
                author = "[deleted]"
            else:
                author = str(submission.author)

            # 处理 crosspost
            crosspost_parent = None
            try:
                if hasattr(submission, 'crosspost_parent'):
                    crosspost_parent = submission.crosspost_parent
            except Exception:
                pass

            # 处理 flair
            flair = ""
            if hasattr(submission, 'link_flair_text') and submission.link_flair_text:
                flair = submission.link_flair_text

            post_data = {
                "post_id": submission.id,
                "title": title,
                "selftext": selftext,
                "text_length": len(selftext),
                "title_length": len(title),
                "author": author,
                "subreddit": submission.subreddit.display_name,
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "num_comments": submission.num_comments,
                "created_utc": int(submission.created_utc),
                "url": url,
                "is_self": is_self,
                "has_image": has_image,
                "image_count": 1 if has_image else 0,
                "link_flair_text": flair,
                "crosspost_parent": crosspost_parent,
                "over_18": submission.over_18,
                "spoiler": submission.spoiler,
                "permalink": submission.permalink,
            }

            return post_data

        except Exception as e:
            logger.warning(f"[RedditTools] 解析帖子失败: {e}")
            return None

    # ==================== 用户信息 ====================

    def fetch_user_info(self, username: str) -> Optional[Dict]:
        """获取用户公开信息"""
        if username in ("[deleted]", "[removed]", "AutoModerator"):
            return None

        reddit = self._get_reddit()
        self._rate_limit()

        try:
            redditor = reddit.redditor(username)
            return {
                "username": username,
                "link_karma": redditor.link_karma,
                "comment_karma": redditor.comment_karma,
                "total_karma": redditor.link_karma + redditor.comment_karma,
                "created_utc": int(redditor.created_utc),
                "is_verified": hasattr(redditor, 'verified') and redditor.verified,
                "has_verified_email": redditor.has_verified_email,
            }
        except Exception as e:
            logger.debug(f"[RedditTools] 获取用户 {username} 信息失败: {e}")
            return None

    # ==================== 批量操作 ====================

    def batch_fetch_posts_with_comments(
            self,
            subreddit_name: str,
            posts_limit: int = 100,
            comments_per_post: int = 25,
            comment_depth: int = 3,
            sort: str = "hot",
            time_filter: str = "year",
            image_output_dir: str = None
    ) -> Dict[str, Any]:
        """
        一站式批量采集：帖子 + 评论 + 图片

        Args:
            subreddit_name: subreddit 名称
            posts_limit: 最大帖子数
            comments_per_post: 每帖最大评论数
            comment_depth: 评论最大深度
            sort: 排序方式
            time_filter: 时间过滤
            image_output_dir: 图片保存目录

        Returns:
            完整采集结果
        """
        logger.info(
            f"[RedditTools] 开始批量采集 r/{subreddit_name}: "
            f"最大 {posts_limit} 帖子"
        )

        # 1. 采集帖子
        posts = self.fetch_subreddit_posts(
            subreddit_name, sort=sort,
            limit=posts_limit, time_filter=time_filter
        )

        if not posts:
            return {
                "status": "error",
                "message": f"未从 r/{subreddit_name} 采集到帖子"
            }

        # 2. 采集评论
        post_ids = [p["post_id"] for p in posts]
        comments_map = self.fetch_batch_post_comments(
            post_ids, max_depth=comment_depth,
            limit=comments_per_post, max_workers=2
        )

        # 将评论关联到帖子
        total_comments = 0
        for post in posts:
            pid = post["post_id"]
            post["comments"] = comments_map.get(pid, [])
            post["actual_comment_count"] = len(post["comments"])
            total_comments += len(post["comments"])

        # 3. 下载图片（可选）
        image_stats = {"downloaded": 0, "skipped": 0, "failed": 0}
        if image_output_dir:
            image_stats = self.download_reddit_images(
                posts, image_output_dir
            )

        # 4. 统计
        result = {
            "status": "success",
            "subreddit": subreddit_name,
            "post_count": len(posts),
            "total_comments": total_comments,
            "image_stats": image_stats,
            "posts": posts,
            "unique_authors": len(set(
                p["author"] for p in posts if p["author"] != "[deleted]"
            )),
        }

        logger.info(
            f"[RedditTools] 批量采集完成: "
            f"{len(posts)} 帖子, {total_comments} 评论, "
            f"{image_stats.get('downloaded', 0)} 图片"
        )

        return result

    def save_posts_json(self, posts: List[Dict], output_path: str):
        """保存帖子数据为 JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        logger.info(f"[RedditTools] 已保存 {len(posts)} 帖子到 {output_path}")

    def load_posts_json(self, input_path: str) -> List[Dict]:
        """从 JSON 加载帖子数据"""
        with open(input_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        logger.info(f"[RedditTools] 已加载 {len(posts)} 帖子从 {input_path}")
        return posts

    def get_stats(self) -> Dict:
        """获取采集统计"""
        return {
            "total_requests": self._request_count,
            "last_request_time": self._last_request_time,
        }

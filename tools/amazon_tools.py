"""
Amazon Product 数据采集工具集
处理 McAuley 公开数据集（UCSD/Stanford SNAP 格式）

功能:
- 产品元数据加载和解析
- 评论数据加载和解析
- 产品图片批量下载
- 共买/共视图提取
- 类目层级解析
"""
import os
import re
import json
import gzip
import time
import urllib.request
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


class AmazonToolManager:
    """Amazon Product 数据工具管理器"""

    # 数据集下载 URL (McAuley 2018 数据集)
    BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2"

    CATEGORY_FILES = {
        "Electronics": "meta_Electronics.json.gz",
        "Books": "meta_Books.json.gz",
        "Clothing_Shoes_and_Jewelry": "meta_Clothing_Shoes_and_Jewelry.json.gz",
        "Home_and_Kitchen": "meta_Home_and_Kitchen.json.gz",
        "Sports_and_Outdoors": "meta_Sports_and_Outdoors.json.gz",
        "Toys_and_Games": "meta_Toys_and_Games.json.gz",
        "Movies_and_TV": "meta_Movies_and_TV.json.gz",
        "CDs_and_Vinyl": "meta_CDs_and_Vinyl.json.gz",
        "Grocery_and_Gourmet_Food": "meta_Grocery_and_Gourmet_Food.json.gz",
        "Pet_Supplies": "meta_Pet_Supplies.json.gz",
        "Automotive": "meta_Automotive.json.gz",
        "Software": "meta_Software.json.gz",
    }

    def __init__(self, data_dir: str = "./data/amazon_raw"):
        """
        初始化 Amazon 工具

        Args:
            data_dir: 原始数据存放目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._metadata_cache: Dict[str, Dict[str, Dict]] = {}  # category -> {asin: metadata}
        self._review_cache: Dict[str, Dict[str, List[Dict]]] = {}  # category -> {asin: [reviews]}

    # ==================== 数据加载 ====================

    def load_metadata(self, category: str, file_path: str = None,
                      limit: int = None) -> Dict[str, Dict]:
        """
        加载 Amazon 产品元数据

        McAuley 数据集每行一个 JSON 对象，格式:
        {
            "asin": "B000...",
            "title": "...",
            "category": ["Electronics", "Camera & Photo", "..."],
            "brand": "...",
            "price": 29.99,
            "description": ["..."],
            "imageURL": ["http://..."],
            "imageURLHighRes": ["http://..."],
            "also_buy": ["B000...", ...],
            "also_view": ["B000...", ...],
            "tech1": "...",
            "tech2": "...",
            "feature": ["..."],
            "main_cat": "Electronics",
            "date": "...",
            ...
        }

        Args:
            category: 产品品类名称
            file_path: 直接指定文件路径（覆盖默认）
            limit: 最大加载数量

        Returns:
            {asin: metadata_dict} 映射
        """
        if category in self._metadata_cache:
            logger.info(f"[AmazonTools] 使用缓存的 {category} 元数据")
            return self._metadata_cache[category]

        if file_path is None:
            # 查找数据文件
            possible_paths = [
                os.path.join(self.data_dir, f"meta_{category}.json"),
                os.path.join(self.data_dir, f"meta_{category}.json.gz"),
                os.path.join(self.data_dir, self.CATEGORY_FILES.get(category, f"meta_{category}.json.gz")),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    file_path = p
                    break

        if file_path is None or not os.path.exists(file_path):
            logger.error(
                f"[AmazonTools] 找不到 {category} 元数据文件。"
                f"请从 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ 下载 "
                f"并放置在 {self.data_dir}/"
            )
            return {}

        logger.info(f"[AmazonTools] 加载 {category} 元数据: {file_path}")

        metadata = {}
        count = 0
        opener = gzip.open if file_path.endswith('.gz') else open

        try:
            with opener(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if limit and count >= limit:
                        break
                    try:
                        item = json.loads(line.strip())
                        asin = item.get("asin", "")
                        if asin:
                            metadata[asin] = self._normalize_metadata(item)
                            count += 1
                    except json.JSONDecodeError:
                        continue

            self._metadata_cache[category] = metadata
            logger.info(
                f"[AmazonTools] 加载完成: {len(metadata)} 个 {category} 产品"
            )

        except Exception as e:
            logger.error(f"[AmazonTools] 加载元数据失败: {e}")

        return metadata

    def load_reviews(self, category: str, file_path: str = None,
                     target_asins: Set[str] = None,
                     reviews_per_product: int = 10) -> Dict[str, List[Dict]]:
        """
        加载 Amazon 产品评论

        McAuley 评论数据每行一个 JSON 对象:
        {
            "reviewerID": "A1...",
            "asin": "B000...",
            "overall": 5.0,
            "reviewText": "...",
            "summary": "...",
            "helpful": [5, 10],
            "unixReviewTime": 1386240000,
            "reviewTime": "12 05, 2013",
            "verified": true,
            "reviewerName": "...",
            "vote": 5,
            "style": {...},
            "image": ["http://..."]
        }

        Args:
            category: 产品品类名称
            file_path: 直接指定文件路径
            target_asins: 只加载这些产品的评论
            reviews_per_product: 每个产品最多保留的评论数

        Returns:
            {asin: [review_dicts]}
        """
        cache_key = f"{category}_{reviews_per_product}"
        if cache_key in self._review_cache:
            return self._review_cache[cache_key]

        if file_path is None:
            possible_paths = [
                os.path.join(self.data_dir, f"{category}.json"),
                os.path.join(self.data_dir, f"{category}.json.gz"),
                os.path.join(self.data_dir, f"{category}_5.json.gz"),
                os.path.join(self.data_dir, f"{category}_5.json"),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    file_path = p
                    break

        if file_path is None or not os.path.exists(file_path):
            logger.error(
                f"[AmazonTools] 找不到 {category} 评论文件。"
                f"请从 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ 下载"
            )
            return {}

        logger.info(f"[AmazonTools] 加载 {category} 评论: {file_path}")

        reviews = defaultdict(list)
        opener = gzip.open if file_path.endswith('.gz') else open
        total_scanned = 0

        try:
            with opener(file_path, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    try:
                        total_scanned += 1
                        review = json.loads(line.strip())
                        asin = review.get("asin", "")

                        # 如果指定了目标产品列表，只加载这些
                        if target_asins and asin not in target_asins:
                            continue

                        if asin:
                            reviews[asin].append(
                                self._normalize_review(review)
                            )
                    except json.JSONDecodeError:
                        continue

            # 限制每产品评论数
            for asin in reviews:
                if len(reviews[asin]) > reviews_per_product:
                    # 按帮助度排序取前 N
                    reviews[asin].sort(
                        key=lambda r: r.get("helpful_votes", 0),
                        reverse=True
                    )
                    reviews[asin] = reviews[asin][:reviews_per_product]

            self._review_cache[cache_key] = dict(reviews)
            logger.info(
                f"[AmazonTools] 加载完成: {len(reviews)} 个产品, "
                f"共 {sum(len(r) for r in reviews.values())} 条评论 "
                f"(扫描 {total_scanned} 行)"
            )

        except (EOFError, OSError) as e:
            # 部分 gzip 文件可能截断，但已读取的数据仍可用
            logger.warning(
                f"[AmazonTools] 评论文件可能不完整 ({e}), "
                f"已加载 {len(reviews)} 个产品的评论"
            )
            # 限制每产品评论数
            for asin in reviews:
                if len(reviews[asin]) > reviews_per_product:
                    reviews[asin].sort(
                        key=lambda r: r.get("helpful_votes", 0),
                        reverse=True
                    )
                    reviews[asin] = reviews[asin][:reviews_per_product]
            self._review_cache[cache_key] = dict(reviews)

        except Exception as e:
            logger.error(f"[AmazonTools] 加载评论失败: {e}")

        return dict(reviews)

    # ==================== 产品过滤 ====================

    def filter_products(self, metadata: Dict[str, Dict],
                        min_reviews: int = 0,
                        has_image: bool = True,
                        has_description: bool = True,
                        price_range: Tuple[float, float] = None,
                        limit: int = None) -> Dict[str, Dict]:
        """
        过滤产品

        Args:
            metadata: 产品元数据映射
            min_reviews: 最少评论数（需先加载评论数据标记）
            has_image: 是否必须有图片
            has_description: 是否必须有描述
            price_range: 价格范围 (min, max)
            limit: 最大数量

        Returns:
            过滤后的 {asin: metadata}
        """
        filtered = {}
        for asin, meta in metadata.items():
            if has_image and not meta.get("image_url"):
                continue
            if has_description and not meta.get("description"):
                continue
            if price_range:
                price = meta.get("price", 0)
                if price < price_range[0] or price > price_range[1]:
                    continue

            filtered[asin] = meta

            if limit and len(filtered) >= limit:
                break

        logger.info(
            f"[AmazonTools] 过滤: {len(metadata)} → {len(filtered)} 产品"
        )
        return filtered

    # ==================== 图片下载 ====================

    def download_product_images(self, products: Dict[str, Dict],
                                output_dir: str,
                                max_images_per_product: int = 3,
                                max_workers: int = 5) -> Dict[str, Any]:
        """
        批量下载产品图片

        Args:
            products: {asin: metadata} 映射
            output_dir: 图片保存目录
            max_images_per_product: 每个产品最大图片数
            max_workers: 并发下载数

        Returns:
            下载统计
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "downloaded": 0, "skipped": 0, "failed": 0,
            "total_images": 0, "details": {}
        }

        def download_one(asin: str, meta: Dict) -> Tuple[str, Dict]:
            """下载单个产品的图片"""
            image_urls = meta.get("image_urls_highres", [])
            if not image_urls:
                image_urls = meta.get("image_urls", [])

            if not image_urls:
                return asin, {"status": "skipped", "reason": "no images", "count": 0}

            # 限制图片数
            urls = image_urls[:max_images_per_product]

            product_img_dir = os.path.join(output_dir, asin)
            os.makedirs(product_img_dir, exist_ok=True)

            downloaded = 0
            for i, url in enumerate(urls):
                ext = '.jpg'
                if '.png' in url.lower():
                    ext = '.png'
                elif '.gif' in url.lower():
                    ext = '.gif'

                save_path = os.path.join(product_img_dir, f"{i}{ext}")

                if os.path.exists(save_path):
                    downloaded += 1
                    continue

                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (compatible; DataCrawlerAgent/1.0)'
                        }
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        with open(save_path, 'wb') as f:
                            f.write(response.read())
                    downloaded += 1
                except Exception:
                    pass

            if downloaded > 0:
                return asin, {
                    "status": "downloaded",
                    "count": downloaded,
                    "dir": product_img_dir
                }
            else:
                return asin, {
                    "status": "failed",
                    "count": 0,
                    "reason": "all downloads failed"
                }

        logger.info(
            f"[AmazonTools] 开始下载 {len(products)} 个产品的图片"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_one, asin, meta): asin
                for asin, meta in products.items()
            }
            for future in as_completed(futures):
                asin, detail = future.result()
                results["details"][asin] = detail
                results["total_images"] += detail.get("count", 0)

                if detail["status"] == "downloaded":
                    results["downloaded"] += 1
                elif detail["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1

        logger.info(
            f"[AmazonTools] 图片下载完成: "
            f"成功 {results['downloaded']}, "
            f"跳过 {results['skipped']}, "
            f"失败 {results['failed']}, "
            f"共 {results['total_images']} 张"
        )
        return results

    # ==================== 图结构提取 ====================

    def extract_co_purchase_graph(self, products: Dict[str, Dict]) -> List[Dict]:
        """
        提取共买 (also_bought) 边

        Returns:
            [{source, target, edge_type}]
        """
        edges = []
        for asin, meta in products.items():
            for related_asin in meta.get("also_buy", []):
                if related_asin in products:  # 只保留产品集内的边
                    edges.append({
                        "source": asin,
                        "target": related_asin,
                        "edge_type": "also_bought"
                    })

        logger.info(f"[AmazonTools] 提取共买边: {len(edges)}")
        return edges

    def extract_co_view_graph(self, products: Dict[str, Dict]) -> List[Dict]:
        """
        提取共看 (also_viewed) 边

        Returns:
            [{source, target, edge_type}]
        """
        edges = []
        for asin, meta in products.items():
            for related_asin in meta.get("also_view", []):
                if related_asin in products:
                    edges.append({
                        "source": asin,
                        "target": related_asin,
                        "edge_type": "also_viewed"
                    })

        logger.info(f"[AmazonTools] 提取共看边: {len(edges)}")
        return edges

    def extract_category_tree(self, products: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        提取类目层级树

        Returns:
            {category_name: {level, product_count, parent}}
        """
        categories = defaultdict(lambda: {"count": 0, "level": 0, "parent": None})

        for asin, meta in products.items():
            cat_path = meta.get("category_path", "")
            if not cat_path:
                continue

            parts = [p.strip() for p in cat_path.split("|") if p.strip()]
            for i, cat_name in enumerate(parts):
                cat_key = "|".join(parts[:i + 1])
                categories[cat_key]["count"] += 1
                categories[cat_key]["level"] = i
                categories[cat_key]["name"] = cat_name
                if i > 0:
                    categories[cat_key]["parent"] = "|".join(parts[:i])

        logger.info(f"[AmazonTools] 提取类目: {len(categories)} 个节点")
        return dict(categories)

    def extract_brands(self, products: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        提取品牌信息

        Returns:
            {brand_name: {product_count, products: [asin]}}
        """
        brands = defaultdict(lambda: {"product_count": 0, "products": []})

        for asin, meta in products.items():
            brand = meta.get("brand", "")
            if brand and brand not in ("", "Unknown", "N/A", "No Brand"):
                brands[brand]["product_count"] += 1
                brands[brand]["products"].append(asin)

        logger.info(f"[AmazonTools] 提取品牌: {len(brands)} 个")
        return dict(brands)

    # ==================== 数据规范化 ====================

    def _normalize_metadata(self, raw: Dict) -> Dict:
        """规范化单个产品元数据"""
        # 处理价格
        price = 0.0
        raw_price = raw.get("price")
        if raw_price:
            if isinstance(raw_price, (int, float)):
                price = float(raw_price)
            elif isinstance(raw_price, str):
                price_match = re.search(r'[\d.]+', raw_price.replace(',', ''))
                if price_match:
                    price = float(price_match.group())

        # 处理描述
        description = raw.get("description", "")
        if isinstance(description, list):
            description = " ".join(description)
        description = description.strip()

        # 处理图片 URL
        image_urls = raw.get("imageURL", [])
        image_urls_highres = raw.get("imageURLHighRes", [])
        if isinstance(image_urls, str):
            image_urls = [image_urls] if image_urls else []
        if isinstance(image_urls_highres, str):
            image_urls_highres = [image_urls_highres] if image_urls_highres else []

        # 处理类目
        category = raw.get("category", [])
        if isinstance(category, str):
            category = [category]
        category_path = "|".join(category) if category else ""

        # 处理品牌
        brand = raw.get("brand", "")
        if isinstance(brand, list):
            brand = brand[0] if brand else ""
        brand = str(brand).strip()

        return {
            "asin": raw.get("asin", ""),
            "title": raw.get("title", ""),
            "title_length": len(raw.get("title", "")),
            "description": description,
            "description_length": len(description),
            "price": price,
            "brand": brand,
            "main_cat": raw.get("main_cat", ""),
            "category_path": category_path,
            "image_urls": image_urls,
            "image_urls_highres": image_urls_highres,
            "image_url": image_urls_highres[0] if image_urls_highres else (
                image_urls[0] if image_urls else ""),
            "image_count": len(image_urls_highres) or len(image_urls),
            "also_buy": raw.get("also_buy", []),
            "also_view": raw.get("also_view", []),
            "feature": raw.get("feature", []),
            "tech1": raw.get("tech1", ""),
            "tech2": raw.get("tech2", ""),
            "date": raw.get("date", ""),
        }

    def _normalize_review(self, raw: Dict) -> Dict:
        """规范化单条评论"""
        helpful = raw.get("helpful", [0, 0])
        if isinstance(helpful, list) and len(helpful) >= 2:
            helpful_votes = helpful[0]
            total_votes = helpful[1]
        else:
            helpful_votes = raw.get("vote", 0) or 0
            total_votes = 0

        review_text = raw.get("reviewText", "") or ""

        return {
            "reviewer_id": raw.get("reviewerID", ""),
            "asin": raw.get("asin", ""),
            "rating": float(raw.get("overall", 0)),
            "review_text": review_text,
            "review_text_length": len(review_text),
            "summary": raw.get("summary", ""),
            "helpful_votes": int(helpful_votes),
            "total_votes": int(total_votes),
            "timestamp": int(raw.get("unixReviewTime", 0)),
            "verified": bool(raw.get("verified", False)),
            "reviewer_name": raw.get("reviewerName", ""),
            "review_images": raw.get("image", []),
        }

    # ==================== 保存/加载 ====================

    def save_products_json(self, products: Dict[str, Dict], output_path: str):
        """保存产品数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 转为列表格式便于序列化
        product_list = list(products.values())
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(product_list, f, ensure_ascii=False, indent=2)
        logger.info(f"[AmazonTools] 已保存 {len(products)} 个产品到 {output_path}")

    def save_reviews_json(self, reviews: Dict[str, List[Dict]], output_path: str):
        """保存评论数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[AmazonTools] 已保存 {len(reviews)} 个产品的评论到 {output_path}"
        )

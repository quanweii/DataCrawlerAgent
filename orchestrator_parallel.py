"""
并行协调器 — 使用多GPU和多进程加速数据处理
继承 DataCrawlerOrchestrator，批量模式走动态编排
"""
from typing import Dict, Any, Optional
from loguru import logger
import os
from datetime import datetime

from orchestrator import DataCrawlerOrchestrator


class ParallelDataCrawlerOrchestrator(DataCrawlerOrchestrator):
    """
    并行数据爬取协调器

    与父类相同的Agent接口，但批量模式通过动态编排
    自动利用多GPU并行提取和multiprocessing并行处理。
    """

    def __init__(
        self,
        context_manager: Optional[Any] = None,
        num_gpus: int = 8,
        num_cpu_workers: int = 32
    ):
        self.num_gpus = num_gpus
        self.num_cpu_workers = num_cpu_workers
        super().__init__(context_manager)
        logger.info(
            f"ParallelDataCrawlerOrchestrator 初始化: "
            f"{num_gpus} GPUs, {num_cpu_workers} CPU workers"
        )

    def execute_batch_pipeline(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        批量模式 — 注入并行配置后走动态编排
        """
        # 注入并行参数到task_config
        if "parallel" not in task_config:
            task_config["parallel"] = {}

        task_config["parallel"].setdefault("enabled", True)
        task_config["parallel"].setdefault("num_gpus", self.num_gpus)
        task_config["parallel"].setdefault("num_workers", self.num_cpu_workers)
        task_config["parallel"].setdefault("batch_size", 20)

        if "parallel_extraction" not in task_config:
            task_config["parallel_extraction"] = True
        if "num_gpus" not in task_config:
            task_config["num_gpus"] = self.num_gpus

        return self.execute_pipeline(task_config)


def create_parallel_orchestrator(num_gpus=8, num_cpu_workers=8) -> ParallelDataCrawlerOrchestrator:
    return ParallelDataCrawlerOrchestrator(num_gpus=num_gpus, num_cpu_workers=num_cpu_workers)

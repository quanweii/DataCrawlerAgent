"""
多智能体通信基础设施
提供 Blackboard(共享状态) + MessageBus(消息总线) 实现Agent间通信
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from loguru import logger


@dataclass
class Message:
    """Agent间消息"""
    sender: str
    recipient: str          # agent_name 或 "broadcast"
    msg_type: str           # quality_issue | reprocess_request | status | result
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0       # 0=info, 1=warning, 2=critical

    def __repr__(self):
        return (f"Message({self.sender}->{self.recipient}, "
                f"type={self.msg_type}, priority={self.priority})")


class Blackboard:
    """
    共享状态存储 — 线程安全，按namespace隔离

    命名空间约定:
        downloader/  — 筛选结果、paper_sources
        collector/   — 下载状态、提取结果、multimodal数据
        cleaner/     — 清洗统计、问题报告
        graph/       — 引用边、匹配率
        formatter/   — 数据集路径、最终统计
        global/      — 任务配置、流水线状态
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[Callable]] = {}

    def write(self, namespace: str, key: str, value: Any):
        """写入状态"""
        with self._lock:
            if namespace not in self._data:
                self._data[namespace] = {}
            self._data[namespace][key] = value

        self._notify(namespace, key, value)
        logger.debug(f"[Blackboard] WRITE {namespace}/{key}")

    def read(self, namespace: str, key: str, default: Any = None) -> Any:
        """读取状态"""
        with self._lock:
            ns = self._data.get(namespace, {})
            return ns.get(key, default)

    def read_namespace(self, namespace: str) -> Dict[str, Any]:
        """读取整个命名空间"""
        with self._lock:
            return dict(self._data.get(namespace, {}))

    def delete(self, namespace: str, key: str):
        """删除状态"""
        with self._lock:
            ns = self._data.get(namespace, {})
            ns.pop(key, None)

    def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """获取完整快照（用于日志/调试）"""
        with self._lock:
            return {ns: dict(data) for ns, data in self._data.items()}

    def subscribe(self, namespace: str, callback: Callable):
        """订阅命名空间变更"""
        if namespace not in self._subscribers:
            self._subscribers[namespace] = []
        self._subscribers[namespace].append(callback)

    def _notify(self, namespace: str, key: str, value: Any):
        """通知订阅者"""
        for cb in self._subscribers.get(namespace, []):
            try:
                cb(namespace, key, value)
            except Exception as e:
                logger.warning(f"[Blackboard] 订阅回调异常: {e}")


class MessageBus:
    """
    Agent间消息总线 — 简单的per-agent队列

    消息类型:
        quality_issue      — 下游Agent报告上游数据质量问题
        reprocess_request  — 请求其他Agent重新处理
        status             — Agent状态通知
        result             — Agent结果通知
    """

    def __init__(self):
        self._queues: Dict[str, List[Message]] = {}
        self._lock = threading.Lock()

    def send(self, message: Message):
        """发送消息"""
        with self._lock:
            if message.recipient == "broadcast":
                for agent_name in self._queues:
                    self._queues[agent_name].append(message)
            else:
                if message.recipient not in self._queues:
                    self._queues[message.recipient] = []
                self._queues[message.recipient].append(message)

        logger.debug(
            f"[MessageBus] {message.sender} -> {message.recipient}: "
            f"{message.msg_type} (priority={message.priority})"
        )

    def receive(self, agent_name: str) -> List[Message]:
        """接收所有待处理消息"""
        with self._lock:
            if agent_name not in self._queues:
                return []
            messages = list(self._queues[agent_name])
            self._queues[agent_name] = []
            return messages

    def peek(self, agent_name: str) -> List[Message]:
        """查看待处理消息但不移除"""
        with self._lock:
            return list(self._queues.get(agent_name, []))

    def register_agent(self, agent_name: str):
        """注册Agent到消息总线"""
        with self._lock:
            if agent_name not in self._queues:
                self._queues[agent_name] = []

    def broadcast(self, sender: str, msg_type: str, content: Dict[str, Any],
                  priority: int = 0):
        """广播消息给所有已注册Agent"""
        msg = Message(
            sender=sender,
            recipient="broadcast",
            msg_type=msg_type,
            content=content,
            priority=priority
        )
        self.send(msg)

    def get_pending_count(self, agent_name: str) -> int:
        """获取待处理消息数量"""
        with self._lock:
            return len(self._queues.get(agent_name, []))

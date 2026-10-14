"""
domains.domain_schema — YAML domain config schema, validation, and DomainConfig bridge.

A DomainYamlConfig completely describes a co-attribute MAG domain. It is loaded from YAML
and can be used to dynamically generate DomainConfig, instantiate agents, and run the
full graph-building pipeline — without any hardcoded per-domain entries in the orchestrator.

Usage:
    from domains import load_domain_yaml
    yc = load_domain_yaml("domains/steam/domain.yaml")
    print(yc.domain)  # "steam"
    print(yc.data.id_field)  # "appid"
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import os
import yaml


# ======================== Schema dataclasses ========================

@dataclass
class RelationConfig:
    """单个 co_attribute 关系定义。"""
    name: str                     # "shared_genre"
    group_by: str                 # "genres" — item 中的分组字段
    cap: int = 100                # 组内最大全连接数
    freq_filter: Optional[float] = None  # 高频组过滤阈值 (e.g. 0.30)
    floor: int = 2                # 组内最少节点数
    field_slice: Optional[str] = None    # 可选: 字段值切片 (e.g. ":3" for TMDb actors)


@dataclass
class DataConfig:
    """数据 schema 描述。"""
    id_field: str = ""            # item 唯一标识字段 (e.g. "appid")
    label_field: str = ""         # 标签字段 (e.g. "genres[0]", "style")
    min_class_size: int = 20      # 小类过滤阈值
    max_items: int = 10000        # 最大 item 数
    data_source: str = "file"     # "file" | "api"
    data_file: str = ""           # JSON 数据文件路径
    id_sanitize: Optional[str] = None  # 可选: ID 清洗正则 (wikiart 模式)


@dataclass
class OutputDirs:
    """输出目录配置。支持 {domain} 占位符。"""
    raw: str = "./data/{domain}_raw"
    items: str = "./data/{domain}_cslo"
    graphs: str = "./data/{domain}_graphs"
    datasets: str = "./data/{domain}_datasets"


@dataclass
class AgentOverride:
    """Agent 类覆盖 (escape hatch for custom domains)。"""
    collector_class: str = ""      # fully qualified Python class path
    collector_agent_name: str = "" # blackboard/registry name
    builder_agent_name: str = ""
    tool_manager_class: str = ""   # e.g. "tools.steam_tools.SteamToolManager"


@dataclass
class ProtocolConfig:
    """Real provenance required for temporal and group-inductive views."""
    time_field: str = ""
    group_field: str = ""
    group_semantics: str = "primary entity group"
    protocols: List[str] = field(default_factory=lambda: [
        "transductive", "split_disjoint", "temporal", "group_inductive"])


@dataclass
class DomainYamlConfig:
    """顶层 YAML 域配置 — 完全描述一个 co-attribute MAG 域。"""
    domain: str = ""               # 域名称 (e.g. "steam")
    graph_mode: str = "homogeneous"
    data: DataConfig = field(default_factory=DataConfig)
    relations: List[RelationConfig] = field(default_factory=list)
    output_dirs: OutputDirs = field(default_factory=OutputDirs)
    agents: AgentOverride = field(default_factory=AgentOverride)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    collector_prompt: str = ""     # 可选: 覆盖默认 collector prompt
    builder_prompt: str = ""       # 可选: 覆盖默认 builder prompt


# ======================== Defaults ========================

DEFAULT_DOMAIN_CONFIG: Dict[str, Any] = {
    "graph_mode": "homogeneous",
    "data": {
        "min_class_size": 20,
        "max_items": 10000,
        "data_source": "file",
        "data_file": "",
        "id_sanitize": None,
    },
    "output_dirs": {
        "raw": "./data/{domain}_raw",
        "items": "./data/{domain}_cslo",
        "graphs": "./data/{domain}_graphs",
        "datasets": "./data/{domain}_datasets",
    },
}


# ======================== Validation ========================

def validate_domain_config(config: dict) -> List[str]:
    """验证 YAML 配置 dict，返回错误列表。空列表 = 合法。"""
    errors = []

    # Required top-level
    if not config.get("domain"):
        errors.append("缺少必填字段: domain")
    if not config.get("data") or not isinstance(config.get("data"), dict):
        errors.append("缺少必填字段: data (dict)")
    else:
        data = config["data"]
        if not data.get("id_field"):
            errors.append("缺少必填字段: data.id_field")
        if not data.get("label_field"):
            errors.append("缺少必填字段: data.label_field")

    # Relations
    rels = config.get("relations", [])
    if not isinstance(rels, list):
        errors.append("relations 必须是 list")
    else:
        for i, rel in enumerate(rels):
            if not isinstance(rel, dict):
                errors.append(f"relations[{i}] 必须是 dict")
                continue
            if not rel.get("name"):
                errors.append(f"relations[{i}]: 缺少 name")
            if not rel.get("group_by"):
                errors.append(f"relations[{i}] ({rel.get('name', '?')}): 缺少 group_by")
            if rel.get("cap") is not None and rel["cap"] < 1:
                errors.append(f"relations[{i}] ({rel.get('name', '?')}): cap 必须 > 0")
            if rel.get("freq_filter") is not None and not (0 < rel["freq_filter"] <= 1):
                errors.append(f"relations[{i}] ({rel.get('name', '?')}): freq_filter 必须在 (0,1]")

    # Output dirs
    od = config.get("output_dirs", {})
    if isinstance(od, dict):
        for key in ["raw", "items", "graphs", "datasets"]:
            if key in od and not isinstance(od[key], str):
                errors.append(f"output_dirs.{key} 必须是 string")

    protocol = config.get("protocol", {})
    # Backward compatibility: legacy YAML without a protocol block still
    # loads, but cannot claim temporal/group-inductive availability.
    requested = protocol.get("protocols", [])
    if "temporal" in requested and not protocol.get("time_field"):
        errors.append("protocol.time_field 是 temporal view 的必填 provenance")
    if "group_inductive" in requested and not protocol.get("group_field"):
        errors.append("protocol.group_field 是 group_inductive view 的必填 provenance")

    return errors


# ======================== YAML Loader ========================

def _interpolate_domain(template: str, domain: str) -> str:
    """替换路径中的 {domain} 占位符。"""
    return template.replace("{domain}", domain)


def load_domain_yaml(path: str) -> DomainYamlConfig:
    """从 YAML 文件加载并验证域配置。

    Args:
        path: YAML 文件路径

    Returns:
        验证后的 DomainYamlConfig

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 验证失败
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"域配置文件不存在: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"YAML 顶层必须是 dict, 收到 {type(raw)}")

    # 验证
    errors = validate_domain_config(raw)
    if errors:
        raise ValueError("域配置验证失败:\n  " + "\n  ".join(errors))

    domain = raw["domain"]

    # 构建 DataConfig
    data_raw = raw.get("data", {})
    data = DataConfig(
        id_field=data_raw.get("id_field", ""),
        label_field=data_raw.get("label_field", ""),
        min_class_size=data_raw.get("min_class_size", 20),
        max_items=data_raw.get("max_items", 10000),
        data_source=data_raw.get("data_source", "file"),
        data_file=data_raw.get("data_file", ""),
        id_sanitize=data_raw.get("id_sanitize"),
    )

    # 构建 relations
    relations = []
    for r in raw.get("relations", []):
        relations.append(RelationConfig(
            name=r["name"],
            group_by=r["group_by"],
            cap=r.get("cap", 100),
            freq_filter=r.get("freq_filter"),
            floor=r.get("floor", 2),
            field_slice=r.get("field_slice"),
        ))

    # 构建 output_dirs (插值 {domain})
    od_raw = raw.get("output_dirs", {})
    output_dirs = OutputDirs(
        raw=_interpolate_domain(od_raw.get("raw", "./data/{domain}_raw"), domain),
        items=_interpolate_domain(od_raw.get("items", "./data/{domain}_cslo"), domain),
        graphs=_interpolate_domain(od_raw.get("graphs", "./data/{domain}_graphs"), domain),
        datasets=_interpolate_domain(od_raw.get("datasets", "./data/{domain}_datasets"), domain),
    )

    # 构建 agent overrides
    ag_raw = raw.get("agents", {})
    agents = AgentOverride(
        collector_class=ag_raw.get("collector_class", ""),
        collector_agent_name=ag_raw.get("collector_agent_name", f"{domain}_collector"),
        builder_agent_name=ag_raw.get("builder_agent_name", f"{domain}_graph_builder"),
        tool_manager_class=ag_raw.get("tool_manager_class", ""),
    )

    proto_raw = raw.get("protocol", {})
    protocol = ProtocolConfig(
        time_field=proto_raw.get("time_field", ""),
        group_field=proto_raw.get("group_field", ""),
        group_semantics=proto_raw.get("group_semantics", "primary entity group"),
        protocols=proto_raw.get("protocols", []),
    )

    return DomainYamlConfig(
        domain=domain,
        graph_mode=raw.get("graph_mode", "homogeneous"),
        data=data,
        relations=relations,
        output_dirs=output_dirs,
        agents=agents,
        protocol=protocol,
        collector_prompt=raw.get("collector_prompt", ""),
        builder_prompt=raw.get("builder_prompt", ""),
    )


# ======================== Bridge to orchestration ========================

def yaml_to_domain_config(yc: DomainYamlConfig) -> "DomainConfig":
    """将 DomainYamlConfig 转换为 orchestrator_agent.DomainConfig。

    这是 YAML 域和现有 OrchestratorAgent 之间的桥接。
    YAML 域不需要修改 DOMAIN_CONFIGS 硬编码字典即可被调度。
    """
    from agents.orchestrator_agent import DomainConfig  # lazy import to avoid circular

    domain = yc.domain
    collector_name = yc.agents.collector_agent_name or f"{domain}_collector"
    builder_name = yc.agents.builder_agent_name or f"{domain}_graph_builder"

    return DomainConfig(
        domain=domain,
        agents_desc={
            collector_name: f"加载/{yc.data.data_source} {domain} 元数据",
            "schema_discovery": "字段画像、schema提议/确认、确定性验证和关系来源登记",
            builder_name: f"构建 {domain} 同构图",
            "dataset_formatter": "将同构节点和边封装为MAG数据集JSON",
            "graph_verifier": "Protocol-aware integrity audit + four protocol views",
        },
        default_plan_factory="_default_plan_co_attribute",
        critical_agents={collector_name, "schema_discovery"},
        task_builders={
            collector_name: "_build_task_co_attribute_collector",
            "schema_discovery": "_build_task_schema_discovery",
            builder_name: "_build_task_co_attribute_graph",
            "dataset_formatter": "_build_task_dataset_formatter",
            "graph_verifier": "_build_task_verifier",
        },
        namespace_map={
            collector_name: f"{domain}_collector",
            "schema_discovery": "schema",
            builder_name: f"{domain}_graph",
            "dataset_formatter": "formatter",
            "graph_verifier": "verifier",
        },
        graph_mode=yc.graph_mode,
    )


def yaml_to_graph_config(yc: DomainYamlConfig) -> Dict[str, Any]:
    """将 DomainYamlConfig 转换为 build_graph_from_config 所需的 config dict。"""
    config = {
        "id_field": yc.data.id_field,
        "label_field": yc.data.label_field,
        "min_class": yc.data.min_class_size,
        "build_order": "filter-then-build",  # default; can be overridden
        "node_type": "item",
        "relations": [],
    }
    if yc.data.id_sanitize:
        config["id_sanitize"] = yc.data.id_sanitize

    for r in yc.relations:
        rel = {"name": r.name, "group_by": r.group_by, "cap": r.cap}
        if r.freq_filter is not None:
            rel["freq_filter"] = r.freq_filter
        if r.floor != 2:
            rel["floor"] = r.floor
        if r.field_slice:
            rel["field_slice"] = r.field_slice
        config["relations"].append(rel)
    return config


def yaml_to_task_config(yc: DomainYamlConfig) -> Dict[str, Any]:
    """Create the executable orchestrator task carried by a domain YAML.

    Keeping this conversion in one place prevents the CLI, schema auditor and
    verifier from silently using different label/relation/protocol definitions.
    """
    graph_config = yaml_to_graph_config(yc)

    def base_field(ref: str) -> str:
        return ref.split("[", 1)[0].split(".", 1)[0]

    label_base = base_field(yc.data.label_field)
    relation_lineage = {
        relation.name: {
            "source_fields": [relation.group_by],
            "target_derived": base_field(relation.group_by) == label_base,
        }
        for relation in yc.relations
    }
    return {
        "domain": yc.domain,
        "graph_mode": yc.graph_mode,
        "data_source": yc.data.data_source,
        "data_file": yc.data.data_file,
        "id_field": yc.data.id_field,
        "id_sanitize": yc.data.id_sanitize,
        "max_items": yc.data.max_items,
        "min_class_size": yc.data.min_class_size,
        "load_from_files": yc.data.data_source == "file",
        "output_dirs": {
            "raw": yc.output_dirs.raw,
            "items": yc.output_dirs.items,
            "graphs": yc.output_dirs.graphs,
            "datasets": yc.output_dirs.datasets,
        },
        "dataset_name": f"{yc.domain}_multimodal_graph_agent",
        "output_format": "json",
        "include_metadata": True,
        "schema_mode": "confirm",
        "schema_hint": graph_config,
        "integrity": {
            "entity_key": yc.data.id_field,
            "relation_lineage": relation_lineage,
        },
        "protocol": {
            "protocols": list(yc.protocol.protocols),
            "time_field": yc.protocol.time_field,
            "group_field": yc.protocol.group_field,
            "group_semantics": yc.protocol.group_semantics,
        },
        "split_seed": 1,
    }

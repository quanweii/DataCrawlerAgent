"""Production schema discovery and deterministic validation agent.

The LLM may propose a schema for a new file-backed domain. Correctness does not
depend on that proposal: field existence, identifier uniqueness, label quality,
relation degeneracy and target lineage are checked deterministically before the
schema is published to the graph builder.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from .react_agent import ReActAgent
from .messages import Blackboard, MessageBus


def _base_field(field_ref: str) -> str:
    return re.split(r"[\[\.]", field_ref or "", maxsplit=1)[0]


def _resolve(row: dict, field_ref: str):
    current: Any = row
    for token in re.split(r"\.(?![^\[]*\])", field_ref or ""):
        match = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", token)
        if not match or not isinstance(current, dict):
            return None
        current = current.get(match.group(1))
        if match.group(2) is not None:
            if not isinstance(current, list):
                return None
            index = int(match.group(2))
            current = current[index] if index < len(current) else None
    return current


def profile_fields(items: List[dict]) -> List[dict]:
    fields = sorted({key for item in items[:1000] for key in item})
    profile = []
    for field in fields:
        values = [item.get(field) for item in items]
        nonempty = [value for value in values if value not in (None, "", [], {})]
        profile.append({
            "name": field,
            "coverage": len(nonempty) / max(len(items), 1),
            "n_unique": len({str(value) for value in nonempty}),
            "dtype": ("list" if nonempty and isinstance(nonempty[0], list)
                      else "number" if nonempty and isinstance(nonempty[0], (int, float))
                      else "string"),
            "examples": [str(value)[:80] for value in nonempty[:3]],
        })
    return profile


def validate_schema(items: List[dict], proposal: Dict[str, Any],
                    min_class_size: int = 20) -> Dict[str, Any]:
    fields = {key for item in items[:1000] for key in item}
    id_field = proposal.get("id_field", "")
    label_field = proposal.get("label_field", "")
    errors, warnings = [], []
    ids = [_resolve(item, id_field) for item in items]
    if _base_field(id_field) not in fields:
        errors.append(f"id_field does not exist: {id_field}")
    elif any(value in (None, "") for value in ids):
        errors.append("id_field contains missing values")
    elif len({str(value) for value in ids}) != len(ids):
        errors.append("id_field is not unique")

    labels = [_resolve(item, label_field) for item in items]
    label_counts = Counter(str(value) for value in labels if value not in (None, "", []))
    kept_classes = sum(count >= min_class_size for count in label_counts.values())
    if _base_field(label_field) not in fields:
        errors.append(f"label_field does not exist: {label_field}")
    elif not 2 <= kept_classes <= 100:
        errors.append(
            f"label_field has {kept_classes} classes meeting min_class_size={min_class_size}; expected 2..100")

    relations, lineage = [], {}
    for relation in proposal.get("relations", []):
        name, group_by = relation.get("name", ""), relation.get("group_by", "")
        if not name or _base_field(group_by) not in fields:
            errors.append(f"invalid relation declaration: {name or '<unnamed>'}/{group_by}")
            continue
        groups = Counter()
        for item in items:
            value = _resolve(item, group_by)
            values = value if isinstance(value, list) else [value]
            groups.update(str(v) for v in values if v not in (None, "", []))
        shared = sum(count for count in groups.values() if count >= 2)
        shared_fraction = shared / max(len(items), 1)
        target_derived = _base_field(group_by) == _base_field(label_field)
        if shared_fraction < 0.05:
            warnings.append(f"relation {name} has low shared coverage ({shared_fraction:.3f})")
        normalized = dict(relation)
        relations.append(normalized)
        lineage[name] = {
            "source_fields": [group_by],
            "target_derived": target_derived,
            "shared_fraction": shared_fraction,
        }

    confirmed = dict(proposal)
    confirmed["relations"] = relations
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "confirmed_schema": confirmed,
        "relation_lineage": lineage,
        "excluded_from_benchmark": sorted(
            name for name, value in lineage.items() if value["target_derived"]),
        "label_stats": {
            "classes": len(label_counts),
            "classes_meeting_min": kept_classes,
            "coverage": sum(label_counts.values()) / max(len(items), 1),
        },
    }


class SchemaDiscoveryAgent(ReActAgent):
    max_iterations = 7

    def __init__(self, client, system_prompt, mcp_tool_manager=None,
                 agent_name="schema_discovery", blackboard=None, message_bus=None):
        super().__init__(client=client, system_prompt=system_prompt,
                         mcp_tool_manager=mcp_tool_manager, agent_name=agent_name,
                         blackboard=blackboard, message_bus=message_bus)
        self._current_task: Dict[str, Any] = {}

    def get_available_actions(self) -> List[str]:
        return ["load_items", "profile_fields", "propose_schema", "validate_schema", "publish_schema", "done"]

    def _default_action(self, observation: Dict) -> Dict[str, Any]:
        result = self._accumulated_results
        for key, action in (("items", "load_items"), ("field_profile", "profile_fields"),
                            ("proposal", "propose_schema"), ("schema_audit", "validate_schema"),
                            ("published", "publish_schema")):
            if not result.get(key):
                return {"action": action, "params": {}, "reasoning": f"deterministic phase: {action}"}
        return {"action": "done", "params": {}, "reasoning": "schema confirmed"}

    def is_done(self, observation: Dict) -> bool:
        return bool(self._accumulated_results.get("published"))

    def act(self, action_plan: Dict) -> Dict[str, Any]:
        handlers = {
            "load_items": self.action_load_items,
            "profile_fields": self.action_profile_fields,
            "propose_schema": self.action_propose_schema,
            "validate_schema": self.action_validate_schema,
            "publish_schema": self.action_publish_schema,
        }
        action = action_plan.get("action", "")
        if action == "done":
            return {"status": "success"}
        if action not in handlers:
            return {"status": "error", "message": f"unknown action: {action}"}
        return handlers[action](action_plan.get("params", {}))

    def action_load_items(self, params: Dict) -> Dict:
        collector_ns = self._current_task.get("collector_ns", "")
        items = self.blackboard.read(collector_ns, "items", []) if self.blackboard else []
        if not items and self._current_task.get("data_file"):
            path = Path(self._current_task["data_file"])
            items = json.loads(path.read_text(encoding="utf-8"))
        self._accumulate_result({"items": items})
        return {"status": "success" if items else "error", "item_count": len(items)}

    def action_profile_fields(self, params: Dict) -> Dict:
        profile = profile_fields(self._accumulated_results.get("items", []))
        self._accumulate_result({"field_profile": profile})
        return {"status": "success", "field_count": len(profile)}

    def action_propose_schema(self, params: Dict) -> Dict:
        mode = self._current_task.get("schema_mode", "confirm")
        proposal = self._current_task.get("schema_hint")
        if mode == "discover":
            prompt = (
                "Propose a homogeneous multimodal attributed graph schema as JSON with "
                "id_field, label_field, min_class, and relations[{name,group_by,cap}]. "
                "Use only these profiled fields and do not use the label field as a relation: "
                + json.dumps(self._accumulated_results.get("field_profile", []), ensure_ascii=False))
            response = self.client.chat.completions.create(
                model=self._current_task.get("model") or os.getenv("MODEL_NAME", "qwen-plus"),
                messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1200)
            text = response.choices[0].message.content or ""
            match = re.search(r"\{[\s\S]*\}", text)
            proposal = json.loads(match.group(0)) if match else None
        if not proposal:
            return {"status": "error", "message": "schema proposal unavailable"}
        self._accumulate_result({"proposal": proposal})
        return {"status": "success", "mode": mode}

    def action_validate_schema(self, params: Dict) -> Dict:
        audit = validate_schema(
            self._accumulated_results.get("items", []),
            self._accumulated_results.get("proposal", {}),
            int(self._current_task.get("min_class_size", 20)))
        self._accumulate_result({"schema_audit": audit})
        return {"status": "success" if audit["valid"] else "error", **audit}

    def action_publish_schema(self, params: Dict) -> Dict:
        audit = self._accumulated_results.get("schema_audit", {})
        if not audit.get("valid"):
            return {"status": "error", "message": "cannot publish invalid schema"}
        if self.blackboard:
            self.blackboard.write("schema", "confirmed_graph_config", audit["confirmed_schema"])
            self.blackboard.write("schema", "relation_lineage", audit["relation_lineage"])
            self.blackboard.write("schema", "audit", audit)
        self._accumulate_result({"published": True})
        logger.info("[SchemaDiscovery] confirmed schema published with deterministic audit")
        return {"status": "success", "published": True,
                "excluded_from_benchmark": audit["excluded_from_benchmark"]}

    def execute(self, task: Dict, upstream_data=None) -> Dict:
        self._current_task = task
        if not self.blackboard:
            self.blackboard = Blackboard()
        if not self.message_bus:
            self.message_bus = MessageBus()
        return self.run_react_loop(task, self.blackboard, self.message_bus)

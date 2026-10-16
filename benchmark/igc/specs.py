#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGC specs + 关系清单呈现（命名 / 盲重命名）。

4 个 spec：S1a(命名,去泄露) / S1b(盲重命名,去泄露) / S2(高同质) / S5(全边,sanity)。
盲重命名：relation → rel_A/rel_B/...（按字母序），gloss 退化为中性，
使泄露关系只能通过【测量】(critic 的 per-rel homophily)发现，而非名字/记忆。
"""
from __future__ import annotations

# ---- 每域关系的语义 gloss（命名模式 S1a/S2 用；盲模式 S1b 不用）----
REL_GLOSS: dict[str, dict[str, str]] = {
    "steam": {
        "shared_category": "two games share a product category",
        "shared_genre": "two games share a genre tag",
        "same_publisher": "two games by the SAME publisher (shared corporate entity)",
        "same_developer": "two games by the SAME developer studio (shared entity)",
    },
    "tmdb": {
        "shared_actor": "two movies share a cast member",
        "shared_genre": "two movies share a genre",
        "same_director": "two movies by the SAME director (shared person entity)",
    },
    "wikiart": {
        "same_artist": "two artworks by the SAME artist (shared creator entity)",
        "shared_style": "two artworks share an art style / movement",
    },
}

# ---- Specs ----
SPECS = {
    "S1a": {
        "id": "S1a",
        "blind": False,
        "text": ("Build a node-classification benchmark graph. The graph must be FAIR: "
                 "exclude any edge relation that would trivially reveal the node label — "
                 "specifically relations that connect items sharing the same identity/creator "
                 "entity (e.g. same artist / developer / publisher / director). "
                 "Keep relations that reflect genuine content or categorical similarity."),
        "target": "min_lp_ceiling",
    },
    "S1b": {
        "id": "S1b",
        "blind": True,
        "text": ("Build a node-classification benchmark graph. The graph must be FAIR: "
                 "exclude any edge relation that would trivially reveal the node label via "
                 "shared identity/creator. Keep relations reflecting content/categorical similarity. "
                 "(Relation names are anonymized; you must judge leakage from the measured signal.)"),
        "target": "min_lp_ceiling",
    },
    "S2": {
        "id": "S2",
        "blind": False,
        "text": ("Build a high-homophily graph so that message-passing (GNN) is beneficial. "
                 "Prefer relations that connect items of the SAME class."),
        "target": "max_homophily",
    },
    "S5": {
        "id": "S5",
        "blind": False,
        "text": "Include ALL available edge relations.",
        "target": "all",
    },
}


def blind_map(relations: list[str]) -> dict[str, str]:
    """relation -> rel_A/rel_B/... 按字母序确定性映射。"""
    return {r: f"rel_{chr(ord('A') + i)}" for i, r in enumerate(sorted(relations))}


def inventory_for_prompt(base: dict, spec: dict) -> list[dict]:
    """构造送给 planner 的关系清单。

    命名(S1a/S2)：真名 + gloss + 边数。
    盲(S1b)：rel_A + 中性 gloss + 边数（不含 same_*/shared_* 词、不含同质性数）。
    """
    dom = base["domain"]
    rels = base["all_relations"]
    bmap = blind_map(rels) if spec["blind"] else {}
    out = []
    for r in rels:
        name = bmap.get(r, r) if spec["blind"] else r
        if spec["blind"]:
            gloss = f"connects two items (anonymized relation {name})"
        else:
            gloss = REL_GLOSS.get(dom, {}).get(r, "connects two items")
        out.append({
            "name": name,
            "gloss": gloss,
            "edge_count": int(base["relations"].get(r, 0)),
        })
    return out

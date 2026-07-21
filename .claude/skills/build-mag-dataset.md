---
name: build-mag-dataset
description: |
  Interactively guides users through creating a new multimodal attribute graph (MAG) 
  domain using the DataCrawlerAgent framework. Auto-discovers data fields, helps 
  design co-attribute relations, generates a domain.yaml config, validates it, 
  and runs the full pipeline. Zero Python code required.
category: data-engineering
tags: [mag, dataset, graph, yaml, co-attribute]
---

# Build MAG Dataset

Create a new multimodal attribute graph (MAG) domain for the DataCrawlerAgent framework.

## Overview

This skill takes a user from "I have a JSON file of items" to "I have a complete MAG dataset"
by auto-discovering fields, helping design relations, generating a validated YAML config,
and running the framework. Expected output: a `domains/{name}/domain.yaml` plus the built graph.

## Phase 1: Gather Requirements

Ask the user these questions (one at a time, not all at once):

1. **Domain name**: Short lowercase name (e.g., "spotify", "goodreads", "imdb").
   Used for directory naming and agent registration.

2. **Data file path**: Absolute or relative path to the JSON file containing items.
   Each item = one node. Must be a JSON array of objects.

3. **What is the classification label?** What field in each item holds the class
   (e.g., genre, category, style)? This becomes `label_field`. If the label is
   nested inside a list (e.g., first element of `genres`), note that — the config
   supports `field[0]` syntax.

After getting these three answers, proceed to Phase 2.

## Phase 2: Data Inspection

Read a sample of the data file (first 5 items) and analyze:

```
python -c "
import json
with open('<data_file>') as f:
    items = json.load(f)
print(f'Total items: {len(items)}')
print(f'Item 0 keys: {list(items[0].keys())}')
print()
for i, item in enumerate(items[:5]):
    print(f'--- Item {i} ---')
    for k, v in item.items():
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:120] + '...'
        print(f'  {k}: {type(v).__name__} = {val_str}')
"
```

From the inspection, report to the user:

- **Detected fields** with their types (str, int, list, dict)
- **Best guess for id_field**: The field that looks like a unique identifier
  (common names: `id`, `_id`, `appid`, `movie_id`, `track_id`, `slug`)
- **Best guess for label_field**: The field that looks like a category
  (common names: `genre`, `category`, `style`, `label`, `type`, `class`)
- **Potential relation fields**: List-type fields with repeated values across items
  (these are natural candidates for `shared_X` relations)
- **Data quality notes**: Any empty lists, null values, deeply nested structures

Ask the user to confirm or correct:
- `id_field`
- `label_field` (with optional `[N]` indexing for list fields)
- `min_class_size` (default: 20)
- `max_items` (default: 10000, or the full file size if smaller)
- Whether ID sanitization is needed (regex pattern for IDs with special chars)

## Phase 3: Relation Design

For each list-type field discovered in Phase 2, ask the user whether to create
a co-attribute relation. Explain the options:

```
Relation: shared_<concept>  (e.g., shared_genre, same_artist)
  group_by: <field>         (the item field whose values define groups)
  cap: N                    (max edges per group; lower = sparser. Default: 30)
  freq_filter: F            (optional, skip groups covering >F of all items.
                             Use for very common values like " Fiction" in genres.
                             Typical: 0.30)
  floor: N                  (min nodes per group; default: 2)
  field_slice: ":N"         (optional, take only first N values from list fields.
                             Use for fields like "actors" where only top-N matter. e.g. ":3")
```

Guide the user with these heuristics:
- **`cap`**: Start with 30 for fine-grained fields (genres, tags), 100 for coarser ones (developer, artist)
- **`freq_filter`**: Only add if there's a dominant value that would create a near-clique.
  Check by running: `Counter(item['<field>'] for item in items).most_common(5)`
- **`field_slice`**: Only for long lists where only the top entries define meaningful relations (e.g., top-3 actors)
- **`floor`**: Usually leave at 2. Raise to 5+ if very noisy groups exist.

For each confirmed relation, ask for:
1. Relation name (auto-suggest: `shared_<field>` or `same_<field>`)
2. Cap value
3. Whether freq_filter is needed — if unsure, run a quick freq check:

```
python -c "
import json
from collections import Counter
with open('<data_file>') as f:
    items = json.load(f)
vals = []
for item in items:
    v = item.get('<field>', [])
    if isinstance(v, list):
        vals.extend(v)
    elif v:
        vals.append(v)
c = Counter(vals)
print(f'Unique values: {len(c)}')
print(f'Total occurrences: {len(vals)}')
for v, n in c.most_common(10):
    print(f'  {v}: {n} ({n/len(items)*100:.1f}% coverage)')
"
```

If any value covers >30% of items, suggest a `freq_filter` to skip it.

## Phase 4: Generate domain.yaml

Write the YAML config to `domains/<domain>/domain.yaml`. Create the directory
if it doesn't exist. Use the exact format from existing domains:

```yaml
# <Domain> Co-Attribute Graph — YAML Domain Config
#
# Auto-generated by build-mag-dataset skill.
# Run with: python main.py --domain-config domains/<domain>/domain.yaml

domain: <domain>
graph_mode: homogeneous

data:
  id_field: <id_field>
  label_field: <label_field>
  min_class_size: <N>
  max_items: <N>
  data_source: file
  data_file: <path_to_json>
  # id_sanitize: '<regex>'  # uncomment if needed

relations:
  - name: <rel_name>
    group_by: <field>
    cap: <N>
    freq_filter: <N>   # optional
    floor: <N>          # optional
    field_slice: ":N"   # optional

output_dirs:
  raw: ./data/<domain>_raw
  items: ./data/<domain>_cslo
  graphs: ./data/<domain>_graphs
  datasets: ./data/<domain>_datasets
```

After writing, validate immediately:

```
python -c "
from domains.domain_schema import load_domain_yaml, yaml_to_graph_config, yaml_to_domain_config
from agents.orchestrator_agent import register_domain_config

yc = load_domain_yaml('domains/<domain>/domain.yaml')
gc = yaml_to_graph_config(yc)
dc = yaml_to_domain_config(yc)
register_domain_config(yc.domain, dc)

print(f'Domain: {yc.domain}')
print(f'ID field: {yc.data.id_field}')
print(f'Label field: {yc.data.label_field}')
print(f'Relations: {len(yc.relations)}')
for r in yc.relations:
    print(f'  - {r.name}: group_by={r.group_by} cap={r.cap}')
print(f'Graph config keys: {list(gc.keys())}')
print('VALIDATION PASSED')
"
```

If validation fails, fix the YAML and re-validate. Do NOT proceed until validation passes.

## Phase 5: Dry-Run Graph Construction

Before running the full orchestrator (which requires API keys and LLM calls),
do a dry-run to verify the graph builds correctly.

**IMPORTANT**: `build_graph_from_config` requires cs_lo directories — each item needs a
`data/<domain>_cslo/<item_id>/<item_id>.md` file (can be minimal dummy content).
If the user's cs_lo data does not yet exist, create minimal dummy cs_lo files first:

```
python -c "
import json, os
with open('<data_file>') as f:
    items = json.load(f)
cs_lo_dir = 'data/<domain>_cslo'
os.makedirs(cs_lo_dir, exist_ok=True)
for item in items:
    iid = str(item['<id_field>'])
    d = os.path.join(cs_lo_dir, iid)
    os.makedirs(d, exist_ok=True)
    # Minimal markdown: just the label
    with open(os.path.join(d, f'{iid}.md'), 'w') as f:
        f.write(f'# {iid}\nLabel: {item.get(\"<label_field>\", \"\")}')
    os.makedirs(os.path.join(d, 'images'), exist_ok=True)
print(f'Created {len(items)} cs_lo dirs at {cs_lo_dir}')
"
```

Then run the dry-run:

```
python -c "
import json
from domains.domain_schema import load_domain_yaml, yaml_to_graph_config
from crawlers.config_driven import build_graph_from_config

yc = load_domain_yaml('domains/<domain>/domain.yaml')
gc = yaml_to_graph_config(yc)
with open(yc.data.data_file) as f:
    items = json.load(f)
nodes, edges, stats = build_graph_from_config(items, yc.output_dirs.items, gc)

print(f'Nodes: {stats[\"node_count\"]:,}')
print(f'Edges: {stats[\"edge_count\"]:,}')
print(f'Classes: {stats[\"num_labels\"]}')
print(f'Homophily: {stats[\"homophily\"]:.3f}')
print(f'Avg degree: {stats[\"avg_degree\"]:.1f}')
print(f'Label distribution:')
for label, count in sorted(stats['label_distribution'].items(), key=lambda x: -x[1]):
    print(f'  {label}: {count}')
print(f'Edge relations:')
for rel, count in stats.get('edge_relation_counts', {}).items():
    print(f'  {rel}: {count:,}')
"
```

Report the stats to the user. Flag any issues:
- **Nodes = 0**: min_class_size or id_field/label_field mismatch — revisit Phase 2
- **Edges = 0**: relation group_by fields may not exist in data — revisit Phase 3
- **Homophily > 0.9**: possible label leakage (label_field is a parent of group_by)
- **Single class dominates >80%**: suggest raising min_class_size or using freq_filter

If the user is satisfied with the stats, proceed to Phase 6.

## Phase 6: Run Full Pipeline (Optional)

Ask the user if they want to run the full orchestrator pipeline. This requires:
- API_KEY and MODEL_BASE_URL set in environment or .env file
- Pre-extracted cs_lo data (text + images) at `data/<domain>_cslo/<item_id>/`

If both conditions are met:
```bash
python main.py --domain-config domains/<domain>/domain.yaml --batch
```

If cs_lo data is not yet extracted, explain:
> The framework expects text and image data at `data/<domain>_cslo/<item_id>/`.
> Each item directory should contain `{item_id}.md` (markdown text) and an `images/`
> subdirectory with image files. This extraction step is domain-specific and requires
> custom tooling. For reference, see how `tools/steam_tools.py` handles Steam data.

## Phase 7: Add cs_lo Extraction (Advanced)

If the user's data source requires API-based extraction (not a pre-existing JSON),
guide them to specify the advanced options in YAML:

```yaml
data:
  data_source: api
  data_file: ""

agents:
  tool_manager_class: tools.<domain>_tools.<Domain>ToolManager
  collector_class: agents.<domain>_collector.<Domain>CollectorAgent  # optional
```

Explain that they need to implement:
1. **ToolManager** (`tools/<domain>_tools.py`): Must expose `load_raw(max_items) -> list[dict]`
2. **cs_lo extraction**: Logic to download/extract text and images per item

Point to `tools/steam_tools.py` as a reference implementation.

## Guardrails

- **NEVER overwrite** an existing `domain.yaml` without asking first. Read it, show the diff, and ask for confirmation.
- **NEVER modify** framework code (orchestrator.py, main.py, orchestrator_agent.py) — the skill only writes YAML files.
- **ALWAYS validate** the YAML after writing it using `load_domain_yaml()`.
- **ALWAYS dry-run** graph construction before declaring success.
- If any step fails, diagnose the error and guide the user through fixing it. Do NOT proceed to the next phase with a broken state.
- For data files > 100MB, sample the first 1000 items for inspection rather than loading the whole file.
- If the data is not a JSON array (e.g., JSONL, CSV), explain the required format and offer to help convert it.

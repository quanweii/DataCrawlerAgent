# Domain Configuration Guide

## Quick Start: Add a New MAG Domain

Create ONE YAML file describing your data schema and relations. No Python code required for standard co-attribute domains.

### Minimal Example (Spotify Music Graph)

```yaml
# domains/spotify/domain.yaml
domain: spotify
graph_mode: homogeneous

data:
  id_field: track_id
  label_field: playlist_genre
  min_class_size: 20
  max_items: 10000
  data_source: file
  data_file: ./data/spotify_raw/spotify_tracks.json

protocol:
  protocols: [transductive, split_disjoint, group_inductive]
  group_field: track_artist
  group_semantics: primary artist

relations:
  - name: same_album
    group_by: track_album_id
    cap: 30
  - name: same_artist
    group_by: track_artist
    cap: 100

output_dirs:
  raw: ./data/spotify_raw
  items: ./data/spotify_cslo
  graphs: ./data/spotify_graphs
  datasets: ./data/spotify_datasets
```

Then run:
```bash
python main.py --domain-config domains/spotify/domain.yaml
```

### Requirements

1. **Data file**: A JSON file containing your items (one item = one node). Each item must have:
   - `id_field`: unique identifier (e.g., "track_id")
   - `label_field`: classification label (e.g., "genres[0]" for first genre)
   - Fields referenced by `relations[].group_by`

2. **cs_lo directory**: Pre-extracted text + images in the standard layout:
   ```
   data/{domain}_cslo/
     {item_id}/
       {item_id}.md        # text content (markdown)
       images/             # image files
         *.jpg
         *.png
   ```

### Schema Reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `domain` | string | Yes | — | Domain name (e.g., "spotify") |
| `graph_mode` | string | No | `homogeneous` | Graph type |
| `data.id_field` | string | Yes | — | Unique ID field in items |
| `data.label_field` | string | Yes | — | Label field path (e.g., "genres[0]", "style") |
| `data.min_class_size` | int | No | 20 | Min items per class |
| `data.max_items` | int | No | 10000 | Max items to load |
| `data.data_source` | string | No | `file` | `file` or `api` |
| `data.data_file` | string | No | "" | Path to JSON data file |
| `data.id_sanitize` | string | No | None | Regex for ID sanitization (wikiart pattern) |

**protocol fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `protocols` | list | Yes | Applicable views chosen from transductive, split_disjoint, temporal, group_inductive |
| `time_field` | string | If temporal | Real timestamp field; never replace it with random order |
| `group_field` | string | If group_inductive | Real primary entity/group field kept wholly within one split |
| `group_semantics` | string | No | Human-readable meaning recorded in the manifest |

The raw MAG may retain a relation whose `group_by` is the target field. The schema auditor records that lineage and the protocol generator excludes that relation from benchmark-derived views. High homophily alone does not cause exclusion.

**relations[] fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | — | Relation name (e.g., "shared_genre") |
| `group_by` | string | Yes | — | Item field to group by |
| `cap` | int | No | 100 | Max edges per group |
| `freq_filter` | float | No | None | Skip groups covering >this fraction |
| `floor` | int | No | 2 | Min nodes per group |
| `field_slice` | string | No | None | Slice multi-value fields (e.g., ":3") |

**output_dirs:**

| Field | Default | Description |
|---|---|---|
| `raw` | `./data/{domain}_raw` | Raw data directory |
| `items` | `./data/{domain}_cslo` | cs_lo extraction directory |
| `graphs` | `./data/{domain}_graphs` | Intermediate graph output |
| `datasets` | `./data/{domain}_datasets` | Final MAG dataset output |

**agents (optional overrides):**

| Field | Description |
|---|---|
| `collector_agent_name` | Custom collector agent name for blackboard |
| `builder_agent_name` | Custom builder agent name for blackboard |
| `tool_manager_class` | Fully qualified tool manager class (e.g., "tools.steam_tools.SteamToolManager") |
| `collector_class` | Custom collector agent class path |

### Advanced: Custom Data Source (API-based)

If your data requires API crawling (not a pre-existing JSON file), specify a tool manager:

```yaml
data:
  data_source: api
  data_file: ""  # not used for API mode

agents:
  tool_manager_class: tools.my_domain_tools.MyDomainToolManager
```

The tool manager must implement:
```python
class MyDomainToolManager:
    def __init__(self, data_dir: str): ...
    def load_raw(self, max_items: int = 10000) -> list[dict]: ...
```

See `tools/steam_tools.py` for a reference implementation.

### Advanced: Custom Collector Agent

For domains with unique data loading logic, specify a custom collector class:

```yaml
agents:
  collector_class: my_package.my_collector.MyCollectorAgent
```

The class must extend `ReActAgent` and implement the standard ReAct interface. See `agents/steam_game_collector.py` for a reference.

### Advanced: Custom Graph Builder

For non-co-attribute graphs (e.g., citation networks), specify a custom `build_fn`:

```yaml
# Not yet in YAML schema — use Python build_fn parameter for now
```
See `crawlers/arxiv.py` for a reference of a non-co-attribute builder.

### Existing Domains

| Domain | YAML Config | Nodes | Edges | Classes |
|---|---|---|---|---|
| steam | `domains/steam/domain.yaml` | 2,760 | ~54K | 8 |
| tmdb | `domains/tmdb/domain.yaml` | 13,037 | ~193K | 18 |
| wikiart | `domains/wikiart/domain.yaml` | 21,968 | ~520K | 64 |

These YAML configs are "dogfood" — they prove the framework can describe its own domains.

### Architecture

```
domains/spotify/domain.yaml   (ONE file — user writes this)
         |
         v
  load_domain_yaml()          (validation + parsing)
         |
         v
  yaml_to_domain_config()     (→ Dynamic DomainConfig)
  yaml_to_graph_config()      (→ Config for build_graph_from_config)
         |
         v
  load_orchestrator_from_yaml()  (→ DataCrawlerOrchestrator)
         |
         v
  main.py --domain-config domains/spotify/domain.yaml
         |
         v
  collector → schema_discovery → graph_builder → formatter → verifier
         |
         v
  data/spotify_datasets/spotify_multimodal_graph_agent.json
```

### Comparison: Before vs After

| | Before (Hardcoded) | After (YAML) |
|---|---|---|
| Files to create | 4+ Python files | 1 YAML file |
| Files to modify | 11+ existing files | 0 |
| Python code required | ~400 LOC | 0 LOC (for file-based domains) |
| Reuse of co-attribute build logic | Copy-paste crawler | Automatic via config_driven |
| Domain registration | 11 touch points in orchestrator | Automatic |
| Backward compatibility | — | `--domain steam` still works unchanged |

# Code-Reuse Manifest (Auditable) — C3 Reusability Claim

**Date:** 2026-07-12
**Scope:** Audit the C3 claim in `docs/paper_logic_DnB_FINAL_2026-07-11.md` §2:
> "共享 ~4500 LOC、每域增量 ~280 LOC（94% 复用）；builder 155→31 LOC/域（−80%）；spotify 已零代码建出（无 spotify agent 类）"

**Reviewer attack surface addressed (FINAL §4.5):** *"零代码/94% 复用 但不可审计、spotify 未建"*.

This manifest makes the 94% figure fully auditable by enumerating every file in the shared skeleton and every per-domain incremental file, with line counts. Machine-readable companion: `docs/code_reuse_manifest.csv`.

---

## TL;DR — Verdict

| Sub-claim | Verdict | Recomputed numbers |
|---|---|---|
| **steam/tmdb/wikiart ≈ 94% reuse** | ✅ **Supported, slightly conservative** | 94.6 %–97.2 % under all honest interpretations (shared 5175–8548 SLOC; per-domain 243–298 SLOC) |
| **"每域增量 ~280 LOC"** | ✅ Exact for tmdb/wikiart; steam slightly higher | steam 309, tmdb 276, wikiart 275 raw wc -l (no-tools, original scope). **Reproduced exactly** from 2026-07-07 |
| **arxiv included in the 94%** | ❌ **NO — must be footnoted** | arxiv = 67.7 %–77.6 % (2471 SLOC incremental). Original docs (PAPER_THESIS_v2 §1.3a, 论文逻辑线 §3.3) honestly list arxiv at ~60 %; the FINAL doc's compressed "每域 ~280/94%" phrasing dropped the arxiv exception |
| **spotify zero-code (no spotify agent/crawler class)** | ✅ **Confirmed** | 0 Python LOC; 41-line YAML only. Built by generic `config_driven.build_graph_from_config` |
| **builder 155→31 LOC/domain (−80 %)** | 🟡 **Half-auditable** | 31 LOC/domain post-dedup: **confirmed** (subclasses 27–28 SLOC / 31–32 raw). 209-LOC shared base: **confirmed** (`agents/domain_graph_builder.py` = 205 SLOC / 239 raw). 155-LOC pre-dedup baseline: **NOT re-measurable** — old wrappers removed during the (uncommitted) 2026-07-09 dedup; figure survives only in `PAPER_THESIS_v2.md` §1.3a and the `domain_graph_builder.py` docstring. −80 % = (155−31)/155 is arithmetically consistent |

**Headline reuse (steam/tmdb/wikiart), most-conservative honest interpretation** = {narrow shared skeleton 5175 SLOC} / {5175 + per-domain-with-tools avg 298} = **94.6 %**. This essentially reproduces the claimed 94 %. Under any broader reading of "shared" the number only rises (to 97 %).

---

## Methodology

### Line-count definition
- **Primary metric = SLOC** = non-blank, non-comment Python lines (`grep -cvE '^\s*$|^\s*#'`). Docstrings count as code (standard SLOC convention). Reported per file in the CSV column `loc`.
- **Secondary metric = raw `wc -l`** (column `raw_wc_l`). **The original 2026-07-07 claim was measured with raw `wc -l`**; this is why the per-domain figures (309/276/275) reproduce exactly against `raw_wc_l`, not SLOC. Both are reported so the claim can be bridged.
- Markdown skill lines are reported but **excluded from Python SLOC totals** (the "zero-code / reuse" claim is explicitly about Python LOC).

### Bucket rules
- **shared** = domain-agnostic framework code, reused unchanged across all domains. Three sub-buckets:
  - **S1 — agent/orchestration infra** (orchestrator, base/react agents, messages/blackboard, cleaner, formatter, verifier, entry)
  - **S2 — declarative config-driven engine** (the C3 reusable core: `crawlers/common.py`, `crawlers/config_driven.py`, `agents/domain_graph_builder.py` shared base, `agents/domain_collector.py`, `domains/domain_schema.py`, the `build-mag-dataset` skill)
  - **S3 — encode + evaluation pipeline** (`benchmark/encode_domain.py`, `domain_loader.py`, `dataset.py`, `preprocessor.py`, `splits.py`, NC/LP tasks, GNN models, `run_newdomain_nc.py`, `run_newdomain_lp.py`, `lp_hard_negative_sampler.py`)
- **per_domain** = code that must be written to add ONE new domain. For agent-built domains this is the domain-specific crawler + collector + graph-builder subclass (+ optional tool/extraction glue).
- **excluded** = amazon/reddit legacy builders/collectors/tools (~3569 SLOC). Explicitly excluded from the 4-domain claim per `PAPER_THESIS_v2.md` line 63.

### Two shared-breadth interpretations (both reported)
The codebase has grown since the 2026-07-07 measurement. To be honest we report reuse under two defensible readings of "shared skeleton":
- **Narrow (original 2026-07-07 scope):** the 6 agent-infra files + `common.py` + 7 encode/eval files enumerated in `PAPER_THESIS_v2.md` §1.3a = **5175 SLOC** (6167 raw). This is the closest reproducible analogue of the claimed "~4500".
- **Broad (full auditable framework today):** S1+S2+S3 = **8548 SLOC** (Python only; 10 306 raw).

### Two per-domain scoping choices (both reported)
- **no-tools (original 2026-07-07 swap surface):** crawler + collector + graph-builder subclass only.
- **with-tools:** adds the domain's `tools/<domain>_tools.py` (data-extraction glue). This is the more conservative (lower reuse) reading because tools are unambiguously domain-specific.

> The claimed "94 %" maps to {narrow shared + with-tools} = **94.6 %**, i.e. the most conservative corner of the honest range. We adopt this as the headline.

---

## 1. Shared skeleton — every file

### S1 — Agent / orchestration infrastructure (5352 SLOC)

| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| orchestrator.py | 591 | 734 | top-level pipeline; `load_orchestrator_from_yaml` generic YAML factory |
| orchestrator_parallel.py | 45 | 57 | parallel orchestrator |
| main.py | 311 | 379 | CLI entry; `--domain-config` YAML route bypasses hardcoded DOMAIN_CONFIGS (spotify route) |
| agents/orchestrator_agent.py | 1053 | 1223 | scheduling logic. ⚠️ embeds ~6 legacy `DOMAIN_CONFIGS` blocks (~23 lines each) for arxiv/reddit/amazon/steam/tmdb/wikiart. **spotify bypasses this dict via `register_domain_config`** |
| agents/base_agent.py | 275 | 322 | agent base class |
| agents/react_agent.py | 298 | 376 | ReAct loop |
| agents/messages.py | 136 | 163 | Blackboard + MessageBus |
| agents/data_cleaner.py | 555 | 648 | data-cleaning agent |
| agents/dataset_formatter.py | 1328 | 1582 | MAG JSON formatter |
| agents/graph_verifier.py | 273 | 323 | adversarial verifier (LP-ceiling / homophily) |
| agents/__init__.py | 51 | 73 | package init |
| **S1 subtotal** | **5352** | **6202** | |

### S2 — Declarative config-driven engine / C3 reusable core (1029 SLOC Python + 231-line skill)

| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| crawlers/common.py | 126 | 142 | DensityEdgeBuilder + homophily + stats + filter_small_classes |
| crawlers/config_driven.py | 201 | 256 | **`build_graph_from_config` — the zero-code graph engine used by spotify** |
| crawlers/__init__.py | 14 | 15 | exports build_X_graph + build_graph_from_config |
| agents/domain_graph_builder.py | 205 | 239 | parameterized graph-builder base (post-dedup shared base; docstring records old ~155 LOC/domain wrappers) |
| agents/domain_collector.py | 231 | 276 | parameterized collector base |
| domains/domain_schema.py | 232 | 295 | YAML schema + validation + DomainConfig bridge |
| domains/__init__.py | 20 | 22 | load_domain_yaml export |
| **S2 Python subtotal** | **1029** | **1245** | |
| .claude/skills/build-mag-dataset.md | 231 | 304 | **build-mag-dataset skill (MARKDOWN, not Python; excluded from SLOC totals)** |

### S3 — Encode + evaluation pipeline (2603 SLOC)

| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| benchmark/encode_domain.py | 95 | 119 | encoder runner |
| benchmark/data/domain_loader.py | 250 | 309 | domain→path dicts (data registration); spotify registered as path entries |
| benchmark/data/dataset.py | 208 | 256 | dataset loader |
| benchmark/data/preprocessor.py | 319 | 393 | feature preprocessor |
| benchmark/data/splits.py | 102 | 121 | leak-free splits |
| benchmark/tasks/node_classification.py | 167 | 203 | NC task |
| benchmark/tasks/link_prediction.py | 375 | 476 | LP task |
| benchmark/models/gnn.py | 317 | 381 | GNN models |
| benchmark/run_newdomain_nc.py | 184 | 223 | NC runner |
| benchmark/run_newdomain_lp.py | 297 | 345 | LP runner |
| scripts/lp_hard_negative_sampler.py | 289 | 355 | hard-negative sampler; spotify registered as path entry |
| **S3 subtotal** | **2603** | **3181** | |

### Shared totals

| | SLOC (Python) | raw wc -l |
|---|---:|---:|
| **Broad shared (S1+S2+S3, excl. markdown skill)** | **8548** | **10 306** |
| Narrow shared (original 2026-07-07 scope — see Methodology) | 5175 | 6167 |

---

## 2. Per-domain incremental code — every file

### 2a. Per-domain breakdown

| domain | crawler | collector | graph-builder subclass | tool glue | **total (with tools)** | **total (no tools, orig scope)** | raw wc -l (no tools) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **steam** | 80 | 152 | 27 | 65 | **324** | 259 | 309 |
| **tmdb** | 80 | 128 | 27 | 50 | **285** | 235 | 276 |
| **wikiart** | 77 | 131 | 28 | 48 | **284** | 236 | 275 |
| **arxiv** (legacy path) | 58 | 630¹ | — (A′: formatter) | 290² + 1493³ | **2471** | 2471 | 3230 |
| **spotify** | 0 | 0 | 0 | 0 | **0** | 0 | 0 (41 YAML) |

¹ `agents/paper_collector.py` (762 raw). ² `agents/arxiv_batch_downloader.py`. ³ `agents/reference_graph_builder.py` citation/fuzzy-OCR engine (2037 raw — the E2 LLM-value evidence).

### 2b. Per-domain files (with SLOC / raw wc -l)

#### steam (324 SLOC with tools / 259 without)
| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| crawlers/steam.py | 80 | 101 | build_steam_graph |
| agents/steam_game_collector.py | 152 | 177 | steam collector |
| agents/steam_game_graph_builder.py | 27 | 31 | thin subclass (post-dedup) |
| tools/steam_tools.py | 65 | 80 | SteamToolManager (excluded from original 2026-07-07 swap surface) |

#### tmdb (285 SLOC with tools / 235 without)
| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| crawlers/tmdb.py | 80 | 100 | build_tmdb_graph |
| agents/tmdb_movie_collector.py | 128 | 145 | tmdb collector |
| agents/tmdb_movie_graph_builder.py | 27 | 31 | thin subclass (post-dedup) |
| tools/tmdb_tools.py | 50 | 60 | TmdbToolManager (excluded from original swap surface) |

#### wikiart (284 SLOC with tools / 236 without)
| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| crawlers/wikiart.py | 77 | 95 | build_wikiart_graph + sanitize_pid |
| agents/wikiart_artwork_collector.py | 131 | 148 | wikiart collector |
| agents/wikiart_artwork_graph_builder.py | 28 | 32 | thin subclass (post-dedup) |
| tools/wikiart_tools.py | 48 | 58 | WikiartToolManager (excluded from original swap surface) |

#### arxiv (2471 SLOC) — NOT a 94 % domain; built via legacy agent path, no domain.yaml
| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| crawlers/arxiv.py | 58 | 73 | build_arxiv_graph |
| agents/paper_collector.py | 630 | 762 | paper collector |
| agents/arxiv_batch_downloader.py | 290 | 358 | arxiv batch download |
| agents/reference_graph_builder.py | 1493 | 2037 | citation/fuzzy-OCR engine (E2 LLM value evidence) |

#### spotify — ZERO Python framework code
| file_path | SLOC | raw wc -l | note |
|---|---:|---:|---|
| domains/spotify/domain.yaml | — | 41 | **YAML only**. Built zero-code via build-mag-dataset skill (2026-07-11). Processed by generic `crawlers/config_driven.build_graph_from_config` |

> `domains/` contains **no** `arxiv/domain.yaml` — arxiv uses the legacy hardcoded agent path (its `DomainConfig` is one of the entries embedded in `agents/orchestrator_agent.py::DOMAIN_CONFIGS`).

---

## 3. Recomputed reuse %

Formula (per-domain): **reuse % = shared / (shared + one_domain_incremental)**.

| Shared breadth | Per-domain scope | 3-domain avg incremental | **reuse % (steam/tmdb/wikiart)** |
|---|---|---:|---:|
| **Narrow (5175 SLOC, original scope)** | **with tools (most conservative)** | **298** | **94.6 %** ← headline, reproduces claimed 94 % |
| Narrow (5175 SLOC) | no tools (original swap surface) | 243 | 95.5 % |
| Broad (8548 SLOC, full framework) | with tools | 298 | 96.6 % |
| Broad (8548 SLOC, full framework) | no tools | 243 | 97.2 % |

| Domain | reuse % (narrow shared) | reuse % (broad shared) |
|---|---:|---:|
| steam | 94.6 % / 95.2 %¹ | 96.4 % / 97.1 %¹ |
| tmdb | 94.8 % / 95.6 %¹ | 96.8 % / 97.3 %¹ |
| wikiart | 95.0 % / 95.6 %¹ | 96.8 % / 97.3 %¹ |
| **arxiv** | **67.7 %** | **77.6 %** |
| **spotify** | **100 %** (0 Python incremental) | **100 %** |

¹ with-tools / no-tools.

### Reproduction of the original 2026-07-07 figures
Per-domain raw `wc -l` (no-tools, original swap surface) **reproduces exactly**:
- steam 101 + 177 + 31 = **309** (doc: 309 ✓)
- tmdb 100 + 145 + 31 = **276** (doc: 276 ✓)
- wikiart 95 + 148 + 32 = **275** (doc: 275 ✓)
- Original reuse: 4500 / (4500 + 280) = **94.1 %** ✓

The shared "~4500" from 2026-07-07 is an under-count of today's framework (narrow-scope today = 5175 SLOC / 6167 raw; the framework grew between 2026-07-07 and 2026-07-12, notably `dataset_formatter.py` at 1582 raw and `orchestrator_agent.py` at 1223 raw). The per-domain figures are frozen and exact. Net effect: today's reuse is **≥ the claimed 94 %**, so the claim is conservative/defensible for steam/tmdb/wikiart.

---

## 4. Spotify zero-code verification (the C3 extensibility exhibit)

**Claim:** spotify built with zero Python framework code; no spotify agent/crawler class exists.

**Verification (2026-07-12):**

1. **No spotify class definition anywhere in the codebase:**
   ```
   $ grep -rniE 'class\s+[A-Za-z_]*[Ss]potify' --include='*.py' .
   (no matches)
   ```
2. **No spotify file in `crawlers/`, `benchmark/`, `agents/`, `tools/`:**
   ```
   $ find crawlers benchmark agents tools -iname '*spotify*'
   (only result dirs: benchmark/processed/spotify, benchmark/results/.../lp_spotify_summary.json)
   ```
3. **spotify appears in framework Python only as data-registration / argparse choices, never as logic:**
   - `orchestrator.py:703` — a **docstring example** inside `load_orchestrator_from_yaml` (illustrative)
   - `benchmark/run_newdomain_nc.py:29` — `"spotify"` added to argparse `choices=[...]`
   - `benchmark/run_newdomain_lp.py:37` — same (argparse choices)
   - `benchmark/encode_domain.py:22` — same (argparse choices)
   - `benchmark/data/domain_loader.py:44,46,54,63` — comment + dataset-path / cslo-path / modality-list dict entries
   - `scripts/lp_hard_negative_sampler.py:49` — dataset-path dict entry
4. **spotify is built by the generic engine:** `domains/spotify/domain.yaml` (41 lines) is consumed by `crawlers/config_driven.build_graph_from_config` via `domains/domain_schema.load_domain_yaml`. The `agents:` block in the YAML names `spotify_collector` / `spotify_graph_builder`, which the generic orchestrator instantiates from the parameterized bases — **no subclass file is needed**.
5. **Built artifact exists:** `data/spotify_datasets/spotify_multimodal_graph_agent.json` + NC/LP/leak-profile results present (per FINAL §2 C3: 10000 nodes / 17440 edges / homo 0.741 / text-only).

**Conclusion:** spotify = **0 Python LOC, 41 YAML lines**. Zero-code extensibility **confirmed**.

> Honesty note: "zero-code" means zero **framework** Python. The 41-line YAML is the declarative input, and 6 pre-existing Python files each received a one-token addition (`"spotify"` in an argparse choices list or a data-path dict) to *register* the new domain for the benchmark runners. No new class, no new function, no new branch — consistent with the FINAL doc's "仅注册：5 处 config choice-list 加 'spotify'".

---

## 5. Builder 155 → 31 LOC/domain sub-claim

**Claim** (PAPER_THESIS_v2 §1.3a/b, FINAL §2 C3): the 3 graph-builder wrappers (steam/tmdb/wikiart) were each ~155 LOC of near-duplicate 5-action ReAct boilerplate; after dedup they share a `DomainGraphBuilderAgent` base (209 LOC) and each domain ships only a ~31 LOC subclass (−80 %).

| Part | Status | Evidence |
|---|---|---|
| Post-dedup subclass ≈ 31 LOC/domain | ✅ **Confirmed** | `agents/steam_game_graph_builder.py` = 27 SLOC / **31 raw**; tmdb = 27 / **31**; wikiart = 28 / **32**. Exact match. |
| Shared base ≈ 209 LOC | ✅ **Confirmed (within rounding)** | `agents/domain_graph_builder.py` = 205 SLOC / 239 raw wc -l. The "209" figure matches an earlier version; current is 205 SLOC. |
| Pre-dedup ≈ 155 LOC/domain | 🟡 **Not independently re-measurable** | The pre-dedup wrapper files **no longer exist in the tree**. They were replaced by the shared base + thin subclasses during the **2026-07-09 dedup, which is uncommitted working-tree state** (the shared-base files `agents/domain_graph_builder.py`, `crawlers/config_driven.py`, `domains/domain_schema.py` do not appear in `git log` at all). The 155 baseline survives only in `PAPER_THESIS_v2.md` §1.3a/b and the `domain_graph_builder.py` module docstring ("之前各 ~155 LOC"). |
| −80 % reduction | ✅ Arithmetically consistent | (155 − 31) / 155 = 80.0 %. |

**Honest framing for the paper:** the dedup is real and the post-dedup artefacts are auditable (shared base + 3 thin subclasses totalling 82 SLOC vs. the 3 standalone amazon/reddit-style legacy builders at 496 + 493 SLOC, which demonstrate the pre-dedup pattern concretely). But the specific "155 LOC/domain before" is a design-era figure, not a re-measurable artifact in the current repository; cite it as "documented in the design notes" rather than "re-measured".

---

## 6. Honest flags for the paper

1. **Scope the 94 % to steam/tmdb/wikiart.** arxiv is 68–78 % (citation engine + paper collector dominate). The FINAL doc's compressed line "每域增量 ~280 LOC（94% 复用）" should read "**for the three co-attribute domains (steam/tmdb/wikiart)**; arxiv is 60–78 % due to the citation/fuzzy-OCR subsystem, which is itself the E2 LLM-value evidence." The original source docs (PAPER_THESIS_v2 §1.3a, 论文逻辑线 §3.3) already state this correctly — the compression happened in FINAL.
2. **State the metric.** Report "94 % = shared-skeleton SLOC / (shared + per-domain incremental SLOC)", name the exact shared file set (or cite this manifest), and note that raw `wc -l` gives the same per-domain figures. Auditable.
3. **Distinguish Python from YAML.** "Zero-code" = zero Python framework LOC for spotify; spotify still requires a 41-line YAML (declarative input) and 6 one-token registrations in argparse/data-path lists.
4. **Note the legacy exclusion.** amazon/reddit builders/collectors/tools (~3569 SLOC) are excluded from the 4-domain claim; list them so a reviewer can verify nothing is hidden.
5. **Builder −80 %:** cite post-dedup artefacts (auditable) + design-note baseline (155); do not claim the 155 figure was re-measured in the current tree.

---

## 7. Files

- This manifest: `docs/code_reuse_manifest_2026-07-12.md`
- Machine-readable companion (domain,file_path,loc,raw_wc_l,bucket,notes): `docs/code_reuse_manifest.csv`
- Claim source: `docs/paper_logic_DnB_FINAL_2026-07-11.md` §2 C3, §4.5
- Original measurement: `docs/PAPER_THESIS_v2.md` §1.3a (lines 41–63), `docs/论文逻辑线与实验全景报告_2026-07-09.md` §3.3 (lines 171–177)

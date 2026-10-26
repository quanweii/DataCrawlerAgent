---
name: build-mag-dataset
description: Inspect a JSON source, confirm entity IDs and a scalar target, design co-attribute relations, and generate and trial-build a MAG-Forge domain YAML. Use for adapting the framework to file-backed sources; new APIs and media extraction require source adapters.
---

# Build a MAG dataset

Work from the MAG-Forge repository root. Obtain the source JSON, domain name,
entity ID path and intended prediction target from the request or inspected
records. Reuse information the researcher has already supplied.

1. Run `python scripts/skill_dataset.py inspect --source <file.json>`.
   Review coverage, field types and repeated values. A JSON array of objects is
   required. For a list-valued target, confirm the intended indexed scalar path
   (for example `genres[0]`); never silently replace the research task.
2. Agree on the ID, target, minimum class support and relation source fields.
   Relations support `cap`, `floor`, `freq_filter`, and `field_slice`. Explain
   consequential choices and obtain missing semantic decisions. Relations based
   on the target field must carry lineage and are excluded from benchmark views.
3. Write `domains/<name>/domain.yaml` using `domains/README.md` and the worked
   `examples/skill_demo/domain.yaml`. Preserve an existing declaration unless
   the requested change calls for updating it; retain a recoverable copy.
4. Run `python scripts/skill_dataset.py trial --config domains/<name>/domain.yaml
   --output <trial-directory>`. This invokes the executable schema contract and
   shared builder. Report nodes, edges, classes, relation counts, and any errors.
   The helper creates temporary label-free placeholders solely to exercise graph
   construction; they are never an input to feature encoding or reported results.
5. Correct invalid paths/types or unsupported classes and repeat the trial.
   Successful structural construction is the output of this skill. For a full
   multimodal run, use actual per-entity text/images, the source adapter and
   `python main.py --domain-config domains/<name>/domain.yaml` with the user's
   configured LLM endpoint. See `docs/REPRODUCIBILITY.md` for benchmark evaluation.

Keep generated outputs in the chosen domain/trial directory. Do not write labels
into text intended for encoding or synthesize missing observations. API sources
need a collector and text/image extraction adapter; do not promise that YAML alone
implements acquisition from an unsupported API. See the existing Steam and
Cleveland adapters for the supported interface.

The bundled example contains deliberately synthetic records for an offline
software demonstration; it is not one of the four research datasets.

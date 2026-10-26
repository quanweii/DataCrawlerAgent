# Data provenance and release scope

The four frozen benchmarks use arXiv papers, Steam game records, TMDB movie
records and WikiArt artwork records. The software also retains historical
schema/construction examples from Spotify, Amazon and Reddit; these are not
additional mature multimodal benchmarks.

## Released observations

The resource archives contain entity IDs, labels, graph edges, frozen modality
representations, time/group metadata, protocol tensors and experiment evidence.
They exclude raw paper full text, descriptions, images and API responses.
Public author, developer, director or artist names may occur in the group
provenance used to reproduce the group-inductive protocol. These are public
entity attributes, not a claim that every released field is anonymous or that
the source content contains no personal information.

## Source rights and acquisition

Code is MIT-licensed. Third-party source content retains its applicable rights
and access conditions; see [DATASET_LICENSING_en.md](DATASET_LICENSING_en.md).
This release does not certify historical robots.txt compliance or supply a
blanket permission for new collection, model training or redistribution.
New collection requires the operator's source access, credentials where needed,
and observance of the applicable source terms and request limits.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Evaluation and interpretation

The released protocols distinguish transductive, split-disjoint, temporal and
group-inductive access. The raw graphs remain separate from views that remove
target-derived relations. Temporal results should be read with the released
label-support analysis. Integrity checks record concrete construction and
partition invariants; they do not establish suitability for every downstream
application. Historical diagnostic snapshots and repaired active selections are
identified separately in [README_ANONYMOUS.md](../README_ANONYMOUS.md).

# Frozen feature alignment

The supplied feature files are unchanged source snapshots, aligned by entity ID
by the original loader. arXiv text/formula files have 45,677 rows for a
45,672-node graph; repeated IDs retain the original last-row resolution used by
the loader. The TMDB image file has 13,036 rows; ID 348877 is missing and receives
a zero vector. Paired tasks exclude missing/nonzero-ineligible rows. The release
does not deduplicate, regenerate or impute these frozen observations.

Graph exports retain the original ID order, labels and all edges. They omit raw
titles/descriptions and image paths. Protocol masks and edge tensors, and all
feature files, retain their original source hashes in artifacts/manifest.json.

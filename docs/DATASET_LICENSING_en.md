# Research resource distribution

The code is MIT-licensed. Dataset resources retain the source-specific terms
and attributions of arXiv, Steam, TMDB and WikiArt. This release distributes
structural IDs/labels/edges, frozen representations and evaluation artifacts;
it does not redistribute source descriptions, paper full text, raw API
responses or artwork/storefront image files.

## Obtain the actual resources

All four datasets are included in the archive parts listed in
[`artifacts/README.md`](../artifacts/README.md), with the complete
[`manifest.json`](../artifacts/manifest.json). From the repository root run:

```bash
python scripts/install_artifacts.py --group datasets
python scripts/install_artifacts.py --group evidence
python scripts/verify_release.py
```

The archive contains the graph, labels, 768-dimensional frozen modality vectors,
saved protocols, time/group metadata and integrity provenance for each dataset.
Raw media acquisition is optional for using these fixed features. Re-encoding
or collecting a fresh snapshot requires the appropriate source access and any
API credentials in the user's own environment. Do not infer uniform reuse rights
for all arXiv papers or artwork images from their public availability.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Official source references

- [arXiv licenses](https://info.arxiv.org/help/license/index.html): metadata
  and article content have different licensing rules; article versions retain
  their individual licenses.
- [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/)
  and the applicable API terms govern source access and content use. This
  repository does not assert a general research redistribution exception.
- [TMDB API terms](https://www.themoviedb.org/api-terms-of-use) and
  [official FAQ](https://developer.themoviedb.org/docs/faq) govern API content,
  attribution and use restrictions, including machine-learning uses. This
  repository does not grant independent rights to that content.
- [WikiArt](https://www.wikiart.org/): artwork and page content retain the
  respective rights; omitting raw images does not create a blanket license.

# Leak-Profile Table (C2 headline) — refreshed 2026-07-15

> Authoritative source: `benchmark/results/leak_profile_table.json` (LP section refreshed from mean-fusion re-run).
> Structure metrics (homophily / lp_ceiling / edges_removed) are fusion-independent and unchanged.

| domain | nodes | classes | homophily | LP-ceil orig→pruned | majority | edges_removed |
|---|---|---|---|---|---|---|
| arxiv | 45672 | 4 | 0.933 | 0.673→0.261 | 0.288 | 63778 |
| steam | 2760 | 8 | 0.524 | 0.557→0.504 | 0.504 | 60472 |
| tmdb | 13037 | 18 | 0.239 | 0.413→0.154 | 0.205 | 214582 |
| wikiart | 21968 | 64 | 0.895 | 0.921→0.024 | 0.025 | 584050 |
| spotify | 10000 | 6 | 0.741 | 0.000→0.000 | 0.167 | 0 |

## LP desaturation (mean-fusion, 10 seeds; AUC random / hard / hard_degree)
### arxiv
| model | random | hard | hard_degree | Δdesat |
|---|---|---|---|---|
| mlp | 0.909 | 0.779 | 0.841 | 0.067 |
| gcn | 0.932 | 0.829 | 0.882 | 0.05 |
| gat | 0.928 | 0.822 | 0.878 | 0.05 |
| graphsage | 0.893 | 0.81 | 0.836 | 0.058 |
| gin | 0.938 | 0.874 | 0.919 | 0.019 |

### steam
| model | random | hard | hard_degree | Δdesat |
|---|---|---|---|---|
| mlp | 0.881 | 0.846 | 0.798 | 0.083 |
| gcn | 0.896 | 0.871 | 0.806 | 0.09 |
| gat | 0.86 | 0.828 | 0.786 | 0.074 |
| graphsage | 0.901 | 0.878 | 0.834 | 0.067 |
| gin | 0.862 | 0.822 | 0.729 | 0.133 |

### tmdb
| model | random | hard | hard_degree | Δdesat |
|---|---|---|---|---|
| mlp | 0.848 | 0.783 | 0.7 | 0.147 |
| gcn | 0.898 | 0.856 | 0.782 | 0.116 |
| gat | 0.871 | 0.813 | 0.743 | 0.128 |
| graphsage | 0.893 | 0.834 | 0.767 | 0.127 |
| gin | 0.82 | 0.816 | 0.727 | 0.094 |

### wikiart
| model | random | hard | hard_degree | Δdesat |
|---|---|---|---|---|
| mlp | 0.977 | 0.78 | 0.749 | 0.228 |
| gcn | 0.994 | 0.905 | 0.876 | 0.118 |
| gat | 0.994 | 0.864 | 0.853 | 0.141 |
| graphsage | 0.994 | 0.859 | 0.844 | 0.149 |
| gin | 0.895 | 0.839 | 0.764 | 0.131 |

### spotify
| model | random | hard | hard_degree | Δdesat |
|---|---|---|---|---|
| mlp | 0.973 | 0.942 | 0.968 | 0.006 |
| gcn | 0.983 | 0.965 | 0.981 | 0.003 |
| gat | 0.988 | 0.975 | 0.985 | 0.002 |
| graphsage | 0.983 | 0.974 | 0.981 | 0.003 |
| gin | 0.975 | 0.969 | 0.971 | 0.004 |
| gcnii | 0.945 | 0.94 | 0.945 | 0.0 |

# t-SNE, UMAP, PCA and clustering

1. Choose the population (usually live singlet CD45+ or CD3+) and the markers that define phenotypes. Exclude scatter, time, viability. Default parameters are the compensated fluorescence channels.
2. Downsample: `max_events` is per sample; 5 000 x a few samples is plenty for t-SNE. The call blocks the window while it runs.
3. `dimension_reduction(method="tsne", population="/Live/CD3+", group="All Samples", max_events=5000, seed=1)` creates tSNE_1/tSNE_2 parameters. `plot(x="tSNE_1", y="tSNE_2")` shows them; `overlays` colour by population.
4. `cluster(method="flowsom", population=..., clusters=10)` creates one population per cluster and returns the median expression per cluster: label clusters from that table ("CD4+ CD45RA+ CCR7+ = naive CD4") and rename (`rename_population`).
5. Compare conditions by cluster frequency (`statistics` over the cluster populations), not by looking at the map.
6. Seeds make runs reproducible; always state the seed, event count and parameters used.
7. Computed parameters are saved next to the workspace (.fdw.data). They cannot be exported to a FlowJo .wsp (the warnings say so).

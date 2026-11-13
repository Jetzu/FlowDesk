# Gating strategy: from raw events to populations you can defend

Work top-down and look at numbers before every gate. A gate placed from a picture alone is a guess.

1. `workspace` - what is loaded, which groups exist, are the samples compensated (`compensation` per sample)?
2. Pick one representative sample (a stimulated one, not a blank) and build the gates on it, then extend them to the group.
3. Standard lymphocyte tree (adapt to the panel):
   - Debris/cells: `plot` FSC-A vs SSC-A, polygon around the main cloud, excluding the origin corner.
   - Singlets: FSC-A vs FSC-H (or FSC-W): polygon along the diagonal. `histogram` of the ratio is not needed; the density map shows the band.
   - Live: viability dye negative. `histogram` -> use `suggested_threshold_raw` (the valley between the two peaks) as the upper edge: `create_gate(shape="range", x="Live/Dead", x_range=[null, threshold])`.
   - Lineage: CD3 vs CD19/CD14 etc. Then CD4 vs CD8 inside CD3+ as a *rectangle quadrant* pair or four rectangles.
4. For every gate: create it on the representative `sample=`, check `statistics` (Count, Freq. of Parent) and the plot's `gates_on_plot`, then put it on all samples with `group="All Samples"` (or the experiment group). Compare the per-sample % of parent; an outlier sample usually needs its own gate: `edit_gate(population=..., sample=...)`.
5. Positive/negative boundaries: use an unstained or FMO sample's histogram, take the 99.9th percentile (`histogram` returns percentiles) as the threshold - not the valley - when the population is not clearly bimodal.
6. Name populations by what they are (`CD4+ T cells`), not by the marker boolean. Booleans (`shape="boolean"`, `operation="and"`, `refs=[...]`) for combinations such as CD4+ CD25+ FoxP3+.
7. Finish with `statistics` over the final populations and `save_workspace`. Tell the user which gates are judgement calls and where.

Pitfalls
- Gate values are raw axis units (compensated intensity for `Comp-` parameters), not the 0-1 display coordinates.
- x="CD4" resolves to the compensated detector. If a stain exists on two detectors, the error lists the candidates; name the parameter.
- A gate with an open end uses `null`: `x_range=[4000, null]`.
- `undo` reverts the last change - also one the user made. Check `events` before undoing.

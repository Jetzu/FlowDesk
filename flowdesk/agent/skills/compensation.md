# Compensation

Spillover must be corrected before any gate on a fluorescence axis is meaningful.

1. `compensation(action="list")` - the matrices in the workspace and which each sample uses. Many FCS files carry an acquisition-defined matrix ($SPILLOVER); if the user trusts it, nothing to do.
2. Single-stain controls: put them into one group (`create_group` "Compensation" with the control samples) or load them with `add_samples(folder=..., group="Compensation")`. Controls are recognised by file name vs channel/stain name ("PE", "FITC", "Unstained").
3. `compensation(action="compute", controls_group="Compensation")` - nothing changes yet. Read the matrix and the per-control diagnostics: the primary channel must be the brightest of its control, the positive/negative separation must be clear, an unstained control improves the background. Off-diagonals above ~40 % deserve a comment; negative spillover means a bad control or the wrong gate on it.
4. `compensation(action="apply", controls_group="Compensation", name="Computed 2026-xx", apply_to_group="All Samples")` - computes, stores the matrix and assigns it in one undo step. `replace=true` to overwrite a matrix with the same name.
5. Verify: `plot` a bivariate of two channels that spill into each other (FITC vs PE) for a single-stain control before and after. Populations should line up with the axes, not lean.
6. `compensation(action="assign", matrix="acquisition", samples=[...])` goes back to the file's own matrix for chosen samples.

The same tube type, same voltages: one matrix for the experiment. Different voltages: new controls.

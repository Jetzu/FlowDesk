# Gate problems and how to find them

- Count is 0 or 100 % of parent: the gate is off-scale. `plot` the parent population and read `axes[].range_raw` and `gates_on_plot`; values must be raw units.
- Gate looks right on one sample, wrong on another: shifted staining or a different voltage/compensation. `histogram` of the same parameter on both; adjust that sample's gate only.
- "ambiguous parameter": two detectors carry the stain; use the full parameter name.
- Population disappeared after changing compensation: gates on `Comp-` parameters keep their raw values, so they move relative to the cells. Re-check each fluorescence gate after (re)applying a matrix.
- Weird shape in FSC/SSC: debris or beads; `data_cleaning` shows flow-rate dips and saturated events.
- A polygon/ellipse on a log or logicle axis: vertices are raw intensities, so it bends on screen. Use `plot` to confirm it encloses the cloud.
- Nothing seems to happen in the window: `events` shows what the user did meanwhile; the window may be showing a modal dialog (calls wait until it is closed).
- Undo went too far: `redo`.

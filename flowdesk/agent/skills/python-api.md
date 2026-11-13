# The `python` tool: cheat sheet

Namespace: `controller` (WorkspaceController), `workspace` (also `ws`), `window` (None headless), `np`, `ops` (the tool layer) and the `flowdesk` package. Set `result` to return data; `print` output comes back as `stdout`. Use `undoable=true` whenever the code changes something (one undo step, views refresh).

```python
s = ops.sample("Donor1_Control")            # same name resolution as the tools
cd4 = s.data("Comp-BV421-A")                # one parameter's raw (compensated) values, numpy array
pop = ops.population(s, "/Lymphocytes/Single cells/CD3+")
mask = pop.mask()                           # boolean per event for this population
result = {"n": int(mask.sum()), "median": float(np.median(cd4[mask]))}
```

- `ops.param(sample, "CD4")` resolves a stain or parameter name to the real parameter name.
- `ops.call("statistics", {...})` / `ops.call("create_gate", {...})` run the structured tools in-process.
- Prefer the structured tools; gates made with `create_gate` / `edit_gate` keep counts, undo and the views consistent.
- Large numpy loops are fine; the call blocks the window while it runs, so keep it short.

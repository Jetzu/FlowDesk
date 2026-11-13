# Coming from FlowJo

- Open a FlowJo 10 workspace directly: `open_workspace(path="...wsp")` (FlowDesk reads samples, gates, groups, compensation, keywords). Missing FCS files are reported; point the workspace at the new folder or `add_samples`.
- `export_flowjo_workspace(path=...)` writes a .wsp back; gates on computed parameters (t-SNE axes, clusters) and derived parameters cannot exist in FlowJo and are listed as warnings.
- Statistics: FlowJo's "Freq. of Parent" = `Freq. of Parent`, "Freq. of Total", Median/Mean/Geometric Mean/CV/Percentile(`percentile=`). Counts match to the event.
- Layout editor, tables and batch reports exist in the app; through the agent use `statistics`, `export_statistics` and `plot`.
- Scale: FlowJo biexponential = FlowDesk `logicle`/`biexponential`. `set_scale` changes the display transform; gate values stay in raw units.

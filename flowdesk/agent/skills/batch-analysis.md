# Batch analysis across samples and donors

1. `workspace` - samples with keywords (Donor, Condition, Well ...) and groups. Missing keywords: `set_keyword(samples, name, value)`; then a live group: `create_group(name="Stimulated", criteria=[{"keyword":"Condition","function":"Equals","value":"Stimulated"}])`.
2. Gate once on a representative sample and apply to the group (`group=`); gates created on a group land on its template and every member, and new members inherit them.
3. `statistics(populations=[...], statistics=["Count","Freq. of Parent","Median"], parameters=["CD69"], group="All Samples")` - one row per sample. Frequencies are percent.
4. Check for failures before reporting: a donor whose lymphocyte gate holds half the usual % of parent, a sample with far fewer events, a flow-rate dip (`data_cleaning`). Say so and fix the gate on that sample (`edit_gate(..., sample=...)`) rather than hiding it.
5. Paired comparison (Control vs Stimulated per donor): read both rows per donor from the table, compute the difference yourself, report n and the direction in each donor. Do not state p-values from three donors as proof.
6. `export_statistics(path=...csv)` for the table; `plot` of the key populations for the figure; `save_workspace`.

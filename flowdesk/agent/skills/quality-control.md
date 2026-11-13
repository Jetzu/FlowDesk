# Quality control before analysis

1. `workspace`: event counts per sample (a tube with 20 % of the usual events needs an explanation), keywords, acquisition date.
2. `data_cleaning(group=...)`: unstable flow rate, signal drift over time, saturated events; percent removed per check. `create_population=true` adds a clean population to gate below.
3. Compensation controls: see the `compensation` skill; check the diagnostics.
4. Viability: gate the dye; report % live per sample.
5. Doublets: FSC-A vs FSC-H singlet gate; report % singlets.
6. Report what was excluded and why, per sample, before biological results.

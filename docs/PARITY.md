# FlowJo feature parity review

Status of every FlowJo 10 feature area in FlowDesk. "Same" means the feature
works the way a FlowJo user expects; details and differences are noted.

| Area | FlowJo | FlowDesk | Notes |
|---|---|---|---|
| **Workspace** | Sample/group panes, keyword columns, sort, rename, delete | Same | Keyword columns via *Edit Columns…*; sort by keyword in the Experiment band |
| Open / Save | `.wsp` | `.fdw` (JSON) + `.wsp` import/export | FlowJo opens FlowDesk's exports (verified, see README) |
| Templates | `.wspt` save/apply | `.fdwt` save/apply; `.wspt`/`.wsp` can be applied as templates | *File ▸ Save as Template…* / *Apply Template…* or open a template file |
| Groups | Manual, keyword-based (live), roles, group gating, drag to group | Same | Compensation group auto-populated from `$FIL` |
| Annotate Experiment | Experiment metadata, panel | Same | Experiment band (Workspace tab) |
| Inspect sample | Header/keyword inspector | Same | |
| Add keyword / edit keywords | Yes | Same, plus from file-name regex or CSV | |
| Plate Editor | 96/384-well view, heat-map of a statistic | Same | Wells from `$WELLID`/`WELL ID` or the file name |
| Cytometer settings | Default scales per cytometer | Same | Cytometry band |
| **Graph window** | Pseudocolor/dot/contour/zebra/density/histogram | Same | |
| Gate tools | Rectangle, polygon, ellipse, quadrant, range, bisector, freehand, magnify | Same (zoom = magnify) | Same letter shortcuts |
| Gate editing | Move/resize, rename, delete, tint, lock, NOT gate, copy/paste, apply to group | Same | |
| Axis selection / transform | Axis buttons, `T` dialog (biex width basis, decades), Ctrl+arrows | Same | Linear / biexponential (logicle) / log / arcsinh |
| Ancestry, drill-down by double-click | Yes | Same | |
| Overlays / backgating | Overlay populations, histogram overlays (offset) | Same | Legend drawn on the plot |
| Statistics window | Yes | Same (dock) | |
| Save plot as image / print | PNG/SVG/PDF | Same | |
| **Statistics** | Count, freq., mean, median, geo-mean, mode, SD, CV, robust SD/CV, min, max, percentile | Same set | |
| Boolean gates | AND / OR / NOT | Same | |
| Derived parameters | Ratio, log, sums | Same (ratio/product/sum/difference/log10) | |
| **Compensation** | Matrix editor, apply to group, wizard from controls | Same | Wizard within ~1 point of FACSDiva on real controls |
| **Layout Editor** | Drag plots, overlays, text, arrange, batch, export PDF/PNG/SVG | Same | |
| **Table Editor** | Population×statistic columns, keywords, batch over group, CSV | Same | |
| **Tools: dimensionality reduction** | tSNE, UMAP, PCA (plugins) | Same (openTSNE/sklearn, umap-learn, sklearn PCA) | Results become parameters you can gate on |
| Tools: clustering | FlowSOM, Phenograph, X-shift | FlowSOM-style SOM + metaclustering, k-means | Phenograph/X-shift not included |
| Tools: population comparison | Chi-squared T(x) | Same + KS | |
| Tools: cell cycle | Dean-Jett-Fox, Watson | Dean-Jett-Fox style fit | |
| Tools: proliferation | Division modelling | Same (dye-dilution peaks; division/proliferation index) | |
| Tools: kinetics | Statistic vs time | Same | |
| Tools: concatenate / downsample / export | Yes | Concatenate, export population as FCS/CSV | Downsample via "max events per sample" when concatenating |
| Plugins | Java plugin API | Python modules with `install(window)` | See `flowdesk/ui/main_window.py` `EXTENSION_MODULES` |
| Calibration (MESF beads) | Yes | Not included | |
| Cloud / portal, licensing | FlowJo Portal | Not applicable | |

Platforms: macOS, Windows and Linux (see docs/INSTALL.md); tests run on all
three in CI.

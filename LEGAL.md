# Independence, trademarks and warranty

This file describes how FlowDesk relates to other software. It is a plain-language statement by the authors, not legal
advice. If you distribute FlowDesk commercially, or have a specific concern, have a lawyer review it.

## Independent work

* FlowDesk is an independent, original program written from scratch in Python and Qt.
* It is built from public standards and publications: FCS 2.0 / 3.0 / 3.1, ISAC Gating-ML 2.0, the logicle transform
  (Parks, Roederer and Moore), and the published algorithms behind its analysis tools (t-SNE, UMAP, FlowSOM, Phenograph,
  k-means, cell-cycle and proliferation models, AutoSpill-style compensation refinement).
* It contains no source code, binaries, icons, images or other assets of any other flow cytometry product. No other
  product was disassembled, decompiled or reverse engineered to write it.
* Reading and writing workspace (`.wsp`, `.wspt`) and matrix (`.mtx`) files written by other programs is
  file-format interoperability, so that people can work with their own data. FlowDesk makes no claim that any other
  program will accept, or produce the same numbers from, every file it writes.

## Trademarks

FlowJo is a trademark of Becton, Dickinson and Company (BD). FACSDiva, CytExpert, FlowKit and any other product or
company name mentioned in this repository belong to their respective owners. They are named only to describe file
compatibility or the data a test was run on.

FlowDesk is **not affiliated with, endorsed by or sponsored by** BD or any other maker of flow cytometry software or
instruments. If a rights holder has a concern about a name or description in this repository, open an issue and it
will be changed promptly.

## Warranty

FlowDesk is provided "as is", without warranty of any kind. It is research software, not a medical device, and is not
validated for diagnostic use. Whoever uses a result is responsible for checking it against their own controls and data.

## Agent integration

The optional MCP server lets AI assistants (Claude Code, Codex, Claude Desktop) operate a running FlowDesk. It runs only
on your computer (localhost bridge, token-protected). What the assistant sends to its provider is governed by that
provider's terms; do not let an assistant work on data you are not allowed to share with it.

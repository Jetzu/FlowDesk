"""A small, dependency-free PowerPoint (.pptx) writer.

Only what a layout page needs: slides of a fixed size holding PNG pictures
and text boxes (font size, bold / italic / underline, colour, alignment and
font family). The package is a plain OOXML zip (ECMA-376 PresentationML) with
one slide master, one blank slide layout and a default theme, which
PowerPoint, Keynote, LibreOffice Impress and Google Slides open.

Coordinates are page points (1/72 inch), converted to EMU (12700 per point).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

EMU_PER_PT = 12700
MIN_SLIDE_EMU = 914400  # 1 inch: PowerPoint's limits for a slide side
MAX_SLIDE_EMU = 51206400  # 56 inches

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
NSDECL = f'xmlns:a="{NS_A}" xmlns:r="{NS_R}" xmlns:p="{NS_P}"'


@dataclass
class PptxPicture:
    png: bytes
    x: float
    y: float
    w: float
    h: float
    name: str = "Picture"
    descr: str = ""


@dataclass
class PptxText:
    text: str
    x: float
    y: float
    w: float
    h: float
    font_size: float = 11.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    color: str = "#000000"
    align: str = "left"  # left | center | right | justify
    font_family: str = ""
    name: str = "Text"
    autofit: bool = True


@dataclass
class PptxSlide:
    shapes: list = field(default_factory=list)  # PptxPicture | PptxText
    title: str = ""


def emu(points: float) -> int:
    return int(round(float(points) * EMU_PER_PT))


def _attr(value) -> str:
    return escape(str(value), {'"': "&quot;"})


def _hex(color: str) -> str:
    c = (color or "#000000").strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) == 8:  # #AARRGGBB (Qt's HexArgb)
        c = c[2:]
    try:
        int(c, 16)
    except ValueError:
        return "000000"
    return c[:6].upper().rjust(6, "0")


def _xfrm(x, y, w, h) -> str:
    return f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{max(1, emu(w))}" cy="{max(1, emu(h))}"/></a:xfrm>'


GROUP_HEADER = (
    '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
    '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
)

ALIGN = {"left": "l", "center": "ctr", "centre": "ctr", "right": "r", "justify": "just"}


def _text_xml(shape: PptxText, shape_id: int) -> str:
    size = int(round(min(4000.0, max(1.0, float(shape.font_size))) * 100))
    rpr = f'lang="en-US" sz="{size}" b="{1 if shape.bold else 0}" i="{1 if shape.italic else 0}"'
    if shape.underline:
        rpr += ' u="sng"'
    fill = f'<a:solidFill><a:srgbClr val="{_hex(shape.color)}"/></a:solidFill>'
    font = f'<a:latin typeface="{_attr(shape.font_family)}"/>' if shape.font_family else ""
    algn = ALIGN.get((shape.align or "left").lower(), "l")
    paras = []
    for line in (shape.text or "").split("\n"):
        if line:
            paras.append(f'<a:p><a:pPr algn="{algn}"/><a:r><a:rPr {rpr} dirty="0">{fill}{font}</a:rPr><a:t>{escape(line)}</a:t></a:r></a:p>')
        else:
            paras.append(f'<a:p><a:pPr algn="{algn}"/><a:endParaRPr {rpr} dirty="0"/></a:p>')
    fit = "<a:spAutoFit/>" if shape.autofit else ""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{_attr(shape.name)} {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr>{_xfrm(shape.x, shape.y, shape.w, shape.h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0">{fit}</a:bodyPr><a:lstStyle/>'
        + "".join(paras)
        + "</p:txBody></p:sp>"
    )


def _picture_xml(shape: PptxPicture, shape_id: int, rid: str) -> str:
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{shape_id}" name="{_attr(shape.name)} {shape_id}" descr="{_attr(shape.descr)}"/>'
        '<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
        f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
        f'<p:spPr>{_xfrm(shape.x, shape.y, shape.w, shape.h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
    )


def _rels(entries: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<Relationship Id="{rid}" Type="{typ}" Target="{_attr(target)}"/>' for rid, typ, target in entries)
    return f'{XML_HEAD}<Relationships xmlns="{NS_REL}">{body}</Relationships>'


THEME = (
    XML_HEAD + f'<a:theme xmlns:a="{NS_A}" name="FlowDesk"><a:themeElements>'
    '<a:clrScheme name="FlowDesk"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="1F2937"/></a:dk2><a:lt2><a:srgbClr val="E5E7EB"/></a:lt2>'
    '<a:accent1><a:srgbClr val="2F6FD6"/></a:accent1><a:accent2><a:srgbClr val="D81B1B"/></a:accent2>'
    '<a:accent3><a:srgbClr val="2E9E4A"/></a:accent3><a:accent4><a:srgbClr val="E08A00"/></a:accent4>'
    '<a:accent5><a:srgbClr val="7A3FC2"/></a:accent5><a:accent6><a:srgbClr val="1A9AA6"/></a:accent6>'
    '<a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme>'
    '<a:fontScheme name="FlowDesk"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
    '<a:fmtScheme name="FlowDesk"><a:fillStyleLst>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
    '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>'
    "</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>"
)

_LVL = '<a:lvl1pPr marL="0" algn="l" defTabSz="914400"><a:defRPr sz="1800" kern="1200"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/><a:cs typeface="+mn-cs"/></a:defRPr></a:lvl1pPr>'

MASTER = (
    XML_HEAD + f"<p:sldMaster {NSDECL}><p:cSld><p:bg><p:bgRef idx=\"1001\"><a:schemeClr val=\"bg1\"/></p:bgRef></p:bg>"
    f"<p:spTree>{GROUP_HEADER}</p:spTree></p:cSld>"
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" '
    'accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    f"<p:txStyles><p:titleStyle>{_LVL}</p:titleStyle><p:bodyStyle>{_LVL}</p:bodyStyle><p:otherStyle>{_LVL}</p:otherStyle></p:txStyles>"
    "</p:sldMaster>"
)

LAYOUT = (
    XML_HEAD + f'<p:sldLayout {NSDECL} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{GROUP_HEADER}</p:spTree></p:cSld>'
    "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
)


def write_pptx(path, slides: list[PptxSlide], page_size: tuple[float, float], title: str = "", author: str = "FlowDesk") -> int:
    """Write `slides` (page_size in points) to `path`. Returns the slide count."""
    slides = list(slides) or [PptxSlide()]
    cx = min(MAX_SLIDE_EMU, max(MIN_SLIDE_EMU, emu(page_size[0])))
    cy = min(MAX_SLIDE_EMU, max(MIN_SLIDE_EMU, emu(page_size[1])))
    n = len(slides)
    media: dict[str, str] = {}  # sha1 -> media file name (identical images stored once)
    files: dict[str, str | bytes] = {}

    for k, slide in enumerate(slides, start=1):
        rels = [("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml")]
        shapes = []
        sid = 2
        for shape in slide.shapes:
            if isinstance(shape, PptxPicture):
                digest = hashlib.sha1(shape.png).hexdigest()
                name = media.get(digest)
                if name is None:
                    name = f"image{len(media) + 1}.png"
                    media[digest] = name
                    files[f"ppt/media/{name}"] = shape.png
                rid = f"rId{len(rels) + 1}"
                rels.append((rid, f"{REL}/image", f"../media/{name}"))
                shapes.append(_picture_xml(shape, sid, rid))
            elif isinstance(shape, PptxText):
                shapes.append(_text_xml(shape, sid))
            else:
                continue
            sid += 1
        files[f"ppt/slides/slide{k}.xml"] = (
            XML_HEAD + f"<p:sld {NSDECL}><p:cSld><p:spTree>{GROUP_HEADER}{''.join(shapes)}</p:spTree></p:cSld>"
            "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
        )
        files[f"ppt/slides/_rels/slide{k}.xml.rels"] = _rels(rels)

    slide_ids = "".join(f'<p:sldId id="{255 + k}" r:id="rId{k + 1}"/>' for k in range(1, n + 1))
    files["ppt/presentation.xml"] = (
        XML_HEAD + f'<p:presentation {NSDECL} saveSubsetFonts="1">'
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
        f'<p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="{cx}" cy="{cy}"/><p:notesSz cx="6858000" cy="9144000"/>'
        "</p:presentation>"
    )
    pres_rels = [("rId1", f"{REL}/slideMaster", "slideMasters/slideMaster1.xml")]
    pres_rels += [(f"rId{k + 1}", f"{REL}/slide", f"slides/slide{k}.xml") for k in range(1, n + 1)]
    pres_rels += [
        (f"rId{n + 2}", f"{REL}/presProps", "presProps.xml"),
        (f"rId{n + 3}", f"{REL}/viewProps", "viewProps.xml"),
        (f"rId{n + 4}", f"{REL}/theme", "theme/theme1.xml"),
        (f"rId{n + 5}", f"{REL}/tableStyles", "tableStyles.xml"),
    ]
    files["ppt/_rels/presentation.xml.rels"] = _rels(pres_rels)
    files["ppt/slideMasters/slideMaster1.xml"] = MASTER
    files["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = _rels(
        [("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", f"{REL}/theme", "../theme/theme1.xml")]
    )
    files["ppt/slideLayouts/slideLayout1.xml"] = LAYOUT
    files["ppt/slideLayouts/_rels/slideLayout1.xml.rels"] = _rels([("rId1", f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml")])
    files["ppt/theme/theme1.xml"] = THEME
    files["ppt/presProps.xml"] = XML_HEAD + f"<p:presentationPr {NSDECL}/>"
    files["ppt/viewProps.xml"] = XML_HEAD + f"<p:viewPr {NSDECL}/>"
    files["ppt/tableStyles.xml"] = XML_HEAD + f'<a:tblStyleLst xmlns:a="{NS_A}" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>'
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files["docProps/core.xml"] = (
        XML_HEAD + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(title)}</dc:title><dc:creator>{escape(author)}</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'
    )
    files["docProps/app.xml"] = (
        XML_HEAD + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        f"<Application>FlowDesk</Application><Slides>{n}</Slides></Properties>"
    )
    ct = "application/vnd.openxmlformats-officedocument"
    overrides = [
        ("/ppt/presentation.xml", f"{ct}.presentationml.presentation.main+xml"),
        ("/ppt/slideMasters/slideMaster1.xml", f"{ct}.presentationml.slideMaster+xml"),
        ("/ppt/slideLayouts/slideLayout1.xml", f"{ct}.presentationml.slideLayout+xml"),
        ("/ppt/theme/theme1.xml", f"{ct}.theme+xml"),
        ("/ppt/presProps.xml", f"{ct}.presentationml.presProps+xml"),
        ("/ppt/viewProps.xml", f"{ct}.presentationml.viewProps+xml"),
        ("/ppt/tableStyles.xml", f"{ct}.presentationml.tableStyles+xml"),
        ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
        ("/docProps/app.xml", f"{ct}.extended-properties+xml"),
    ]
    overrides += [(f"/ppt/slides/slide{k}.xml", f"{ct}.presentationml.slide+xml") for k in range(1, n + 1)]
    files["[Content_Types].xml"] = (
        XML_HEAD + f'<Types xmlns="{NS_CT}">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
        + "".join(f'<Override PartName="{part}" ContentType="{typ}"/>' for part, typ in overrides)
        + "</Types>"
    )
    files["_rels/.rels"] = _rels(
        [
            ("rId1", f"{REL}/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", f"{REL}/extended-properties", "docProps/app.xml"),
        ]
    )
    order = ["[Content_Types].xml", "_rels/.rels"] + sorted(k for k in files if k not in ("[Content_Types].xml", "_rels/.rels"))
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in order:
                data = files[name]
                if isinstance(data, str):
                    data = data.encode("utf-8")
                # PNGs are already compressed
                zf.writestr(name, data, compress_type=zipfile.ZIP_STORED if name.endswith(".png") else zipfile.ZIP_DEFLATED)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n

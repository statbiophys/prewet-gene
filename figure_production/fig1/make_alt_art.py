"""Artwork of the rearranged Fig. 1 (alt/), derived from ../../fig1/art_{a,d}.svg.

Provenance record, run once by hand from figure_production/:

    python alt/fig1/make_alt_art.py

It writes, next to itself:
  art_a_alt.svg/.pdf    panel A rearranged: polymer on top (left tail curving
                        down, leftmost monomer removed, right tail curving down
                        and ending south-west), interaction table below it
                        with the two eta pairs on one row (brace removed),
                        transcription model to the table's right. The probe's
                        leader line and the "/" between the eta pairs are drawn
                        by the notebook. Full source canvas, placed 1:1.
  art_d_{free,closed,open}.svg/.pdf
                        one cloud of panel D each (with its glows and dots), on
                        the full source canvas; the notebook places each with
                        its own scale/offset. art_d.svg's opaque gradient stops
                        are kept, so Quartz renders them as poppler does.
  art_alt_layout.json   the positions the notebook needs to set A's text.

The PDFs are what the notebook stamps; nothing is converted at render time.
"""
import copy
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
INKSCAPE = "/Applications/Inkscape.app/Contents/MacOS/inkscape"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# ---- panel A geometry (source-canvas units) -----------------------------------
POLY_D = (-7.0, -63.0)          # the whole chain, labels and bp arrows
CHAIN_V = 78.35                 # chain axis in the source
BEAD = 4.37                     # bead path bbox (diameter)
STEP = 6.4                      # bead spacing along the tails
# left tail: the source's three remaining beads mirrored about the chain axis
LEFT_TAIL = {"path3446-4-8": (20.60, 77.18), "path3446-7-5": (14.55, 74.54),
             "path3446-1-0": (10.65, 68.83)}
REMOVED = ("path3446-41-9",          # leftmost monomer
           "surface1-1-1-4-7",       # the brace grouping the two eta rows
           "path3406-6-9")           # probe leader line (redrawn in the notebook)
RIGHT_TAIL = ("path3446-7-3-0", "path3446-1-8-8", "path3446-41-6-0",
              "path3446-5-0-9", "path3446-18-4-9")
RIGHT_START = (118.35, 78.61)        # centre of the bead after the probe (source)
RIGHT_HEADINGS = (25, 60, 95, 130, 165)   # deg, v downwards: curls down, ends SW

# interaction table: one bead column centred on U_C, rows at ROW_V
U_C = 19.0
ROW_V = (56.0, 65.5, 75.0, 84.5)     # v=, eta, alpha, rho
X_EQ = 43.0                          # the "=" column
TITLE_V = 45.0                       # top of both titles
PAIR_GAP = 6.22                      # one bead diameter between the eta pairs
# transcription model: source centre u 120.7, title top v 8.8
MODEL_U = 95.0
MODEL_D = (MODEL_U - 120.7, TITLE_V - 8.8)


def bbox_centres(svg_path):
    out = subprocess.run([INKSCAPE, svg_path, "--query-all"], check=True,
                         capture_output=True, text=True).stdout
    boxes = {}
    for line in out.splitlines():
        i, x, y, w, h = line.split(",")
        boxes[i] = (float(x), float(y), float(w), float(h))
    return boxes


def wrap(parent, el, dx, dy):
    """Replace el by <g transform=translate(dx,dy)>el</g>, in place."""
    idx = list(parent).index(el)
    parent.remove(el)
    g = ET.Element("{%s}g" % SVG, {"transform": "translate(%.4f,%.4f)" % (dx, dy)})
    g.append(el)
    parent.insert(idx, g)


def to_pdf(svg_path):
    pdf = svg_path[:-4] + ".pdf"
    subprocess.run([INKSCAPE, svg_path, "--export-type=pdf", "--export-area-page",
                    "--export-filename=" + pdf], check=True, capture_output=True)
    return pdf


def build_a():
    src = os.path.join(SRC, "art_a.svg")
    boxes = bbox_centres(src)
    cen = {k: (x + w / 2, y + h / 2) for k, (x, y, w, h) in boxes.items()}
    tree = ET.parse(src)
    art = tree.getroot().find("{%s}g[@id='art']" % SVG)
    els = {e.get("id"): e for e in art}
    for k in REMOVED:
        art.remove(els.pop(k))

    du, dv = POLY_D
    # right tail: a curve leaving the chain eastwards, bending down, ending SW
    u, v = RIGHT_START
    right = {}
    for k, hdg in zip(RIGHT_TAIL, RIGHT_HEADINGS):
        u += STEP * math.cos(math.radians(hdg))
        v += STEP * math.sin(math.radians(hdg))
        right[k] = (u, v)
    table = {"g1684": (U_C, ROW_V[0]),
             "path3446-6-8-7-6-4-8-8": (U_C - 3.185, ROW_V[2]),     # blue/blue, left
             "path3446-6-8-7-6-4-8": (U_C + 3.185, ROW_V[2]),       # blue/blue, right
             "g50": (U_C, ROW_V[3])}
    half = (boxes["g1685"][2] + PAIR_GAP + boxes["g58"][2]) / 2
    table["g1685"] = (U_C - half + boxes["g1685"][2] / 2, ROW_V[1])   # blue/yellow
    table["g58"] = (U_C + half - boxes["g58"][2] / 2, ROW_V[1])       # blue/red
    model = ("path3060-9", "path3075-1", "path3110-2", "path3111-7", "path3115-0",
             "path3116-9", "path3506-2-6", "g31")

    for k, el in list(els.items()):
        if k in table:
            tu, tv = table[k]
            wrap(art, el, tu - cen[k][0], tv - cen[k][1])
        elif k in model:
            wrap(art, el, *MODEL_D)
        elif k in LEFT_TAIL:
            ou, ov = LEFT_TAIL[k]
            tu, tv = ou, 2 * CHAIN_V - ov            # mirror about the chain
            wrap(art, el, tu - cen[k][0] + du, tv - cen[k][1] + dv)
        elif k in right:
            tu, tv = right[k]
            wrap(art, el, tu - cen[k][0] + du, tv - cen[k][1] + dv)
        else:                                        # chain, probe, ligand, arrows
            wrap(art, el, du, dv)
    out = os.path.join(HERE, "art_a_alt.svg")
    tree.write(out, xml_declaration=True, encoding="utf-8")
    to_pdf(out)

    probe = cen["path3506-2"]
    layout = {
        "poly_d": POLY_D, "u_c": U_C, "row_v": ROW_V, "x_eq": X_EQ,
        "title_v": TITLE_V, "model_d": MODEL_D,
        "probe": (probe[0] + du, probe[1] + dv),
        "right_tail": {k: (u + du, v + dv) for k, (u, v) in right.items()},
    }
    return layout


D_PIECES = {"free": "g1799", "closed": "g1810", "open": "g1809"}


def build_d():
    src = os.path.join(SRC, "art_d.svg")
    boxes = bbox_centres(src)
    info = {}
    for name, keep in D_PIECES.items():
        tree = ET.parse(src)
        root = tree.getroot()
        g = root.find(".//{%s}g[@id='g1689']" % SVG)
        for c in list(g):
            if c.get("id") != keep:
                g.remove(c)
        out = os.path.join(HERE, "art_d_%s.svg" % name)
        tree.write(out, xml_declaration=True, encoding="utf-8")
        to_pdf(out)
        x, y, w, h = boxes[keep]
        info[name] = {"centre": (x + w / 2, y + h / 2), "size": (w, h)}
    return info


if __name__ == "__main__":
    layout = build_a()
    layout["d_clouds"] = build_d()
    with open(os.path.join(HERE, "art_alt_layout.json"), "w") as fh:
        json.dump(layout, fh, indent=1)
    print(json.dumps(layout, indent=1))

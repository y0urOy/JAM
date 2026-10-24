#!/usr/bin/env python3
"""Switch off every theme effect (drop shadows) in a PowerPoint figure.

python-pptx's default theme gives all three effect styles a drop shadow, so any shape that does not
carry its own empty <a:effectLst/> is drawn with one in PowerPoint: connectors, freeform flames and
accent strokes, the invisible connector anchors. The figures are flat; this writes the empty list into
every shape's properties, recursing through groups, and points the style reference at effect 0.

    python strip_pptx_effects.py file.pptx [more.pptx ...]
"""
import sys
from lxml import etree
from pptx import Presentation

A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
# what may follow effectLst inside spPr, in schema order
AFTER = {'{%s}%s' % (A, t) for t in ('effectDag', 'scene3d', 'sp3d', 'extLst')}


def flatten(sp_pr):
    """Ensure sp_pr holds exactly one empty effectLst, placed where the schema wants it."""
    for old in sp_pr.findall('{%s}effectLst' % A):
        sp_pr.remove(old)
    lst = etree.SubElement(sp_pr, '{%s}effectLst' % A)
    for i, child in enumerate(sp_pr):
        if child.tag in AFTER:
            sp_pr.remove(lst); sp_pr.insert(i, lst); break


def strip(path):
    prs = Presentation(path); n = 0
    for slide in prs.slides:
        root = slide._element
        for sp_pr in root.iter('{http://schemas.openxmlformats.org/presentationml/2006/main}spPr'):
            flatten(sp_pr); n += 1
        for ref in root.iter('{%s}effectRef' % A):
            ref.set('idx', '0')
    prs.save(path); return n


if __name__ == '__main__':
    for f in sys.argv[1:]:
        print('%s: effects removed from %d shapes' % (f, strip(f)))

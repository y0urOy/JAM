#!/usr/bin/env python3
"""Native draw.io and PowerPoint copies of the paper's Figures 1 and 2.

Run only to regenerate the starting copies. Never run over a hand-edited file.
"""
from __future__ import annotations

import argparse
import sys, os
import re
import base64
import html
import json
import math
from html.parser import HTMLParser
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
import cairocffi as cairo
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
COL = dict(blue='0072B2', green='2F8F5B', orange='CC7A00', orangedark='A35F00', orangelight='E39A2E', gray='7F8C96',
           ink='17212B', muted='5E6C78', panel='C8D4DE', paper='FCFDFE')
SCALE = 100.0  # pixels per original TikZ coordinate (1.5 cm)
CM = SCALE / 1.5
BOX_RADIUS=0.15   # cm, corner radius of module boxes in the PowerPoint export
PT = SCALE / (1.5 * 72 / 2.54)
MARGIN = 14


def mix(hexcolor, fraction):
    return ''.join(f'{round(255+(int(hexcolor[i:i+2],16)-255)*fraction):02X}'
                   for i in (0,2,4))


class RichText(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.lines = [[]]
        self.tags = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.lines.append([])
        else:
            self.tags.append(tag)
            if tag == 's':
                self.small_factor = float(dict(attrs).get('f', 0.8))

    def handle_endtag(self, tag):
        if tag in self.tags:
            self.tags.remove(tag)

    def handle_data(self, data):
        self.lines[-1].append((data, 'b' in self.tags, 'i' in self.tags,
                              -1 if 'sup' in self.tags else 1 if 'sub' in self.tags else 0,
                              'g' in self.tags,     # <g>...</g> = muted grey secondary text
                              getattr(self, 'small_factor', 0.8) if 's' in self.tags else 1.0))   # <s f=0.8>...</s> = smaller text


class Diagram:
    def __init__(self, name, width, bottom, top):
        self.name, self.top = name, top
        self.width, self.height = width*SCALE+2*MARGIN, (top-bottom)*SCALE+2*MARGIN
        self.nodes, self.edges = {}, []

    def node(self, key, x, y, w, h, label='', color='blue', kind='box',
             bold=False, dash=False, fill=None, size=8, image=None):
        color = COL.get(color, color)
        width, height = w*CM, h*CM
        self.nodes[key] = dict(id=key, x=x*SCALE-width/2+MARGIN,
            y=(self.top-y)*SCALE-height/2+MARGIN, w=width, h=height,
            label=label, stroke=color, fill=fill if fill is not None else
            ('FFFFFF' if kind in ('box','encoder') else mix(color,.12)),
            kind=kind, bold=bold, dash=dash, size=size*PT, image=image,
            label_shift=(-.06 if w>=4 else -.05)*CM if kind=='encoder' else 0)
        return key

    def panel(self, key, x0, y0, x1, y1):
        return self.node(key,(x0+x1)/2,(y0+y1)/2,(x1-x0)*1.5,(y1-y0)*1.5,
                         color='panel',kind='panel',fill=COL['paper'])

    def label(self, key, x, y, w, h, text, size=7.5, bold=False, color='muted'):
        return self.node(key,x,y,w,h,text,color,'text',bold,fill='none',size=size)

    def edge(self, source, target, color='blue', start=(1,.5), end=(0,.5),
             arrow=True, dash=False):
        self.edges.append(dict(id=f'edge_{len(self.edges)}',source=source,target=target,
            color=COL[color],start=start,end=end,arrow=arrow,dash=dash))

    def point(self, name, anchor):
        n=self.nodes[name]
        return n['x']+n['w']*anchor[0],n['y']+n['h']*anchor[1]

    def pair(self, key, x,y,left,right,color='blue',dash=False,w=.55,h=.42,gap=.46,size=7,band=.0):
        """Two labelled token boxes with an ellipsis between them: [g_1] ... [g_h].

        The ellipsis is the middle of three children, so a connector arriving from above or below
        attaches at the group's centre, while one arriving from a side attaches to the box there.
        """
        # A band behind the two boxes so the pair reads as one sequence, and so connectors have a
        # border to land on: without it an arrow stops at an invisible edge a box-height from the
        # dots it points at. Created first, so it sits behind its own children.
        # wider than tall: the wash to the sides is what ties the two boxes together, while a deep
        # one would eat the room the arrows need above and below
        pad, vpad = (band, band*0.375) if band else (0.0, 0.0)
        self.node(key,x,y,2*w+gap+2*pad,h+2*vpad,color=color,
                  kind='band' if band else 'group',
                  # a shade lighter than the boxes it holds, so it backs them without
                  # competing; the boxes themselves are mixed at .12
                  fill=mix(COL[color],.05) if band else 'none')
        step=(w+gap)/2/1.5                                  # centimetres to diagram units
        for i,(txt,dx,kind,cw,ch) in enumerate(((left,-step,'token',w,h),
                                                # without a band the ellipsis box is as tall as the tokens, so a
                                                # vertical connector meets the pair level with the boxes' edges
                                                ('···',0.0,'text',gap,.22 if band else h),
                                                (right,step,'token',w,h))):
            child=self.node(f'{key}_{i}',x+dx,y,cw,ch,txt,color=color,kind=kind,dash=dash,
                            size=size,fill='none' if kind=='text' else None)
            self.nodes[child]['parent']=key

    def row(self, key, x,y,count,color='blue',dash=False,pitch=.24):
        width=(count-1)*pitch*SCALE+.30*CM
        self.node(key,x,y,width/CM,.3,color=color,kind='group',fill='none')
        for i in range(count):
            child=self.node(f'{key}_{i}',x+(i-(count-1)/2)*pitch,y,.30,.30,
                           color=color,kind='token',dash=dash)
            self.nodes[child]['parent']=key


def seq(symbol, end='h', noisy=False, predicted=False):
    pre={('g',True,False):'ĝ',('a',False,True):'ã'}
    mark='̃' if noisy else '̂' if predicted else ''
    sym=pre.get((symbol,predicted,noisy),symbol+mark)
    return f'<i>{sym}</i><sub>1</sub> ··· <i>{sym}</i><sub>{end}</sub>'


def loss(sub):
    return f'ℒ<sub>{sub}</sub>'


def build_designspace():
    d=Diagram('Figure 1',11.41,1.62,6.08)
    for prefix,shift,title in [('a',0,'(a) World-action model'),
                               ('b',3.63,'(b) Vision-language-action model')]:
        d.panel(prefix+'Panel',shift,1.62,shift+3.48,6.08)
        d.label(prefix+'Title',shift+1.74,5.85,5.0,.36,title,8,True,'ink')
        d.node(prefix+'F',shift+1.05,5.01,2.7,.5,seq('f',predicted=True)
               if prefix=='a' else seq('f')+': unused',color='blue' if prefix=='a' else 'gray',
               kind='token',dash=prefix=='b',size=7)
        d.node(prefix+'Out',shift+2.793,5.01,1.6,.5,seq('a'),color='green',kind='token',size=7)
        d.node(prefix+'V',shift+1.05,3.97,2.7,.9,
               'Generator<br>backbone' if prefix=='a' else 'VLM<br>backbone',bold=True)
        d.node(prefix+'A',shift+2.793,3.97,1.6,.9,'Action<br>expert',color='green',bold=True)
        d.node(prefix+'N',shift+2.793,2.92,1.6,.5,seq('a',noisy=True),
               color='green',kind='token',dash=True,size=7)
        if prefix=='a':
            d.node('aObs',.35,2.92,.6,.5,'<i>f</i><sub>0</sub>',kind='token',size=7)
            d.node('aNoisy',1.317,2.92,1.9,.5,seq('f',noisy=True),kind='token',dash=True,size=7)
            for name,x in [('aObs',.35),('aNoisy',1.317)]:
                d.edge(name,'aV',start=(.5,0),end=((x-(1.05-.9))/1.8,1))
            d.edge('aV','aF',start=(.5,0),end=(.5,1))
        else:
            d.node('bObs',shift+.55,2.92,.6,.5,'<i>f</i><sub>0</sub>',kind='token',size=7)
            d.node('bText',shift+1.55,2.92,.6,.5,'<i>c</i>',color='gray',kind='token',dash=True,size=7)
            d.edge('bObs','bV',start=(.5,0),end=((.55-.15)/1.8,1))
            d.edge('bText','bV','gray',start=(.5,0),end=((1.55-.15)/1.8,1))
        d.edge(prefix+'V',prefix+'A')
        d.edge(prefix+'N',prefix+'A','green',start=(.5,0),end=(.5,1))
        d.edge(prefix+'A',prefix+'Out','green',start=(.5,0),end=(.5,1))
        d.node(prefix+'VisLoss',shift+1.05,2.35,2.7,.36,
            loss('G')+': generate the future' if prefix=='a' else 'no future loss',
            color='blue' if prefix=='a' else 'gray',kind='token',dash=prefix=='b',size=7)
        d.node(prefix+'ActLoss',shift+2.793,2.35,1.6,.36,loss('act'),color='green',kind='token',size=7)
    s=7.26
    d.panel('cPanel',s,1.62,s+4.15,6.08)
    d.label('cTitle',s+2.075,5.85,6.0,.36,'(c) JAM: one recipe bridging both families',8,True,'ink')
    d.node('cE',s+1.35,5.35,3.5,.55,'Frozen target encoder',color='orange',bold=True,dash=True,fill=mix(COL['orange'],.05))
    d.node('cFuture',s+1.35,4.73,1.6,.5,'<i>f</i><sub>0</sub> '+seq('f'),color='gray',kind='token',size=7)
    d.node('cH',s+1.35,4.2,3.5,.55,'Prediction head',color='orange',bold=True,fill=mix(COL['orange'],.12))
    d.node('cV',s+1.35,3.0,3.5,.85,'<b>Pretrained backbone</b><br><g>Generative DiT · SSL · VLM</g>',size=7.4)
    d.node('cObs',s+1.35,2.3,.6,.5,'<i>f</i><sub>0</sub>',kind='token',size=7)
    d.node('cVisLoss',s+1.35,1.85,3.5,.36,loss('F')+': predict the future',color='orange',kind='token',size=7)
    d.node('cG',s+3.45,5.35,1.6,.5,seq('g'),color='orange',kind='token',size=7)
    d.node('cP',s+3.45,4.2,1.6,.5,seq('g',predicted=True),color='orange',kind='token',dash=True,size=7)
    d.node('cLoss',s+3.45,4.775,.38,.38,loss('F'),color='orange',kind='circle',size=7)
    d.node('cOut',s+3.45,3.7,1.6,.5,seq('a'),color='green',kind='token',size=7)
    d.node('cA',s+3.45,3,1.6,.85,'Action<br>expert',color='green',bold=True)
    d.node('cNoise',s+3.45,2.3,1.6,.5,seq('a',noisy=True),color='green',kind='token',dash=True,size=7)
    d.node('cActLoss',s+3.45,1.85,1.6,.36,loss('act'),color='green',kind='token',size=7)
    d.edge('cFuture','cE','gray',start=(.5,0),end=(.5,1))
    d.edge('cE','cG','orange'); d.edge('cH','cP','orange')
    d.edge('cG','cLoss','orange',start=(.5,1),end=(.5,0),arrow=False,dash=True)
    d.edge('cLoss','cP','orange',start=(.5,1),end=(.5,0),arrow=False,dash=True)
    d.edge('cObs','cV',start=(.5,0),end=(.5,1))
    d.edge('cV','cH','orange',start=(.5,0),end=(.5,1))
    d.label('cHidden',s+1.64,3.60,.70,.36,'<i>h</i><sup>\u2009(ℓ)</sup>',8,color='orange')
    d.edge('cV','cA'); d.edge('cNoise','cA','green',start=(.5,0),end=(.5,1))
    d.edge('cA','cOut','green',start=(.5,0),end=(.5,1))
    return d


def build_method():
    d=Diagram('Figure 2',12.45,1.35,5.35)
    d.panel('trainingPanel',0,1.35,9.6,5.35); d.panel('deploymentPanel',9.72,1.35,12.45,5.35)
    d.label('trainingTitle',8.9,5.13,1.65,.34,'Training',8,True,'ink')
    d.label('deploymentTitle',11.57,5.13,2.17,.34,'Deployment',8,True,'ink')
    d.label('instruction',1.15,5.05,3.15,.32,'<i>“Put the bowl on the plate”</i>',7.5,color='ink')
    for key,x,label in [('text',1.15,'Text encoder'),('input',3.45,'Input encoder')]:
        d.node(key,x,4.62,2.64,.5,label,color='gray',kind='encoder',bold=True,fill=mix(COL['gray'],.12))
        d.label(key+'Snow',x+.67,4.62,.25,.3,'❄',9,color='blue')
        d.nodes[key+'Snow']['parent']=key
    d.node('observation',5.097,4.62,1.15,1.15,kind='image',image='spatial_current.png')
    d.label('obsLabel',5.097,5.12,.4,.30,'<i>x</i><sub>t</sub>',7.5,color='blue')
    d.edge('observation','input',start=(0,.5),end=(1,.5))
    d.row('textTokens',1.15,3.98,3,'gray'); d.row('visualTokens',3.45,3.98,1)
    d.label('f0Label',3.20,3.98,.32,.3,'<i>f</i><sub>0</sub>',7.5,color='blue')
    d.edge('text','textTokens','gray',start=(.5,1),end=(.5,0))
    d.edge('input','visualTokens',start=(.5,1),end=(.5,0))
    d.node('backbone',3.115,3.17,7.1,1.0,'<b>Pretrained backbone</b><br><g>Generative DiT · SSL · VLM</g>',fill=mix(COL['blue'],.10),size=7.5)
    d.node('action',6.85,3.17,2.8,1.0,'Action expert',color='green',bold=True,fill=mix(COL['green'],.10))
    d.edge('textTokens','backbone','gray',start=(.5,1),end=((1.15-(3.115-7.1/3))/(7.1/1.5),0))
    d.edge('visualTokens','backbone',start=(.5,1),end=((3.45-(3.115-7.1/3))/(7.1/1.5),0))
    d.edge('backbone','action')
    d.label('kv',5.71,3.42,.42,.26,'<b>KV</b>',7.5,color='blue')
    d.row('state',6.3,4.22,1,'green'); d.row('noisyActions',7.2,4.22,3,'green',True)
    d.label('stateLabel',6.3,4.52,.7,.3,'state',7.5,color='green')
    d.label('noiseLabel',7.2,4.52,1.7,.3,'<i>ã</i><sub>t</sub> ··· <i>ã</i><sub>t+H</sub>',7.5,color='green')
    for key,x in [('state',6.3),('noisyActions',7.2)]:
        d.edge(key,'action','green',start=(.5,1),end=((x-(6.85-2.8/3))/(2.8/1.5),0))
    d.row('actionTokens',8.45,3.17,3,'green')
    d.label('actTokenLabel',8.45,2.87,1.7,.3,'<i>a</i><sub>t</sub> ··· <i>a</i><sub>t+H</sub>',7.5,color='green')
    d.edge('action','actionTokens','green')
    d.node('actionLoss',9.2,3.17,.5,.5,loss('act'),color='green',kind='circle',size=7.5)
    d.edge('actionTokens','actionLoss','green',arrow=False,dash=True)
    d.node('head',1.15,1.88,2.7,.85,'Prediction<br>head <i>p</i>',color='orange',bold=True,fill=mix(COL['orange'],.12))
    d.edge('backbone','head','orange',start=((1.15-(3.115-7.1/3))/(7.1/1.5),1),end=(.5,0))
    d.label('hidden',1.43,2.43,.7,.36,'<i>h</i><sup>\u2009(ℓ)</sup>',7.5,color='orange')
    d.row('predTokens',2.74,1.88,3,'orange',True); d.row('targetTokens',4.22,1.88,3,'orange')
    d.label('predLabel',2.74,2.18,1.3,.3,seq('g',predicted=True),7.5,color='orange')
    d.label('targetLabel',4.22,2.18,1.3,.3,seq('g'),7.5,color='orange')
    d.edge('head','predTokens','orange')
    d.node('foresightLoss',3.48,1.88,.44,.44,loss('F'),color='orange',kind='circle',size=7.5)
    d.edge('predTokens','foresightLoss','orange',arrow=False,dash=True)
    d.edge('foresightLoss','targetTokens','orange',arrow=False,dash=True)
    d.node('teacher',6.565,1.88,5.0,.8,'<b>Frozen target encoder <i>E</i><sub>φ</sub></b><br><g>Generative DiT · SSL · VLM · VAE</g>',color='orange',kind='encoder',size=7.2,fill=mix(COL['orange'],.12))
    d.label('teacherSnow',7.98,1.88,.3,.3,'❄',9,color='blue')
    d.nodes['teacherSnow']['parent']='teacher'
    d.edge('teacher','targetTokens','orange',start=(0,.5),end=(1,.5))
    for key,x,y in [('futureBack',9.01,2),('futureMiddle',8.95,1.94),('future',8.89,1.88)]:
        d.node(key,x,y,1,1,color='orange',kind='image',image='spatial_recorded_t8.png')
    d.label('futureLabel',8.96,2.5,1.0,.3,'<i>x</i><sub>t+1:T</sub>',7.5,color='orange')
    d.edge('future','teacher','orange',start=(0,.5),end=(1,.5))
    for key,x,y in [('backboneFire',5.30,3.17),('actionFire',7.60,3.17),('headFire',1.86,1.88)]:
        d.node(key,x,y,.20,.34,color='orange',kind='flame',fill=COL['orange'])
        d.nodes[key]['parent']=key.removesuffix('Fire')
    d.label('deployInstruction',11.085,4.9,3.65,.28,'<i>“Put the bowl on the plate”</i>',7.5,color='ink')
    d.node('deployObs',11.085,4.42,.9,.9,kind='image',image='spatial_current.png')
    d.node('deployBackbone',11.085,3.46,3.65,.85,'<b>Pretrained backbone</b><br><g>Generative DiT · SSL · VLM</g>',size=7.5,fill=mix(COL['blue'],.10))
    d.node('deployAction',11.085,2.52,3.65,.85,'Action expert',color='green',bold=True,fill=mix(COL['green'],.10))
    d.row('deployActions',11.085,1.76,3,'green')
    d.label('deployActionLabel',11.085,1.51,1.8,.3,'<i>a</i><sub>t</sub> ··· <i>a</i><sub>t+H</sub>',7.5,color='green')
    d.edge('deployObs','deployBackbone',start=(.5,1),end=(.5,0))
    d.edge('deployBackbone','deployAction',start=(.5,1),end=(.5,0))
    d.label('deployKV',11.34,2.99,.4,.26,'<b>KV</b>',7.5,color='blue')
    d.edge('deployAction','deployActions','green',start=(.5,1),end=(.5,0))
    return d


def build_designspace_v2():
    """Figure 1 (a) VLA, (b) WAM, (c) JAM, driven by Figures/fig1_layout.json (centimetres)."""
    spec=json.loads((ROOT/'Figures/fig1_layout.json').read_text()); spec.pop('_comment',None)
    u=lambda cm: cm/1.5
    S=lambda k: spec[k]
    panels=[S(k) for k in ('aPanel','bPanel','cPanel')]
    d=Diagram('Figure 1',u(max(p['x1'] for p in panels)),u(min(p['y0'] for p in panels)),u(max(p['y1'] for p in panels)))
    def panel(key):
        e=S(key); d.panel(key,u(e['x0']),u(e['y0']),u(e['x1']),u(e['y1']))
    def lab(key,text,size=7,bold=False,color='muted'):
        e=S(key); d.label(key,u(e['x']),u(e['y']),e['w'],e['h'],text,size,bold,color)
    def box(key,text,color='blue',size=9,bold=True,dash=False,fill=None):
        e=S(key); d.node(key,u(e['x']),u(e['y']),e['w'],e['h'],text,color=color,bold=bold,dash=dash,size=size,
                         fill=fill if fill is not None else ('FFFFFF' if not dash else mix(COL[color],.05)))
    def tok(key,text,color='blue',dash=False,size=9,bold=False,fill=None):
        e=S(key); d.node(key,u(e['x']),u(e['y']),e['w'],e['h'],text,color=color,kind='token',dash=dash,size=size,bold=bold,fill=fill)
    def frac(key,parent):  # horizontal position of `key` along the top/bottom edge of `parent`
        e,pe=S(key),S(parent); return (e['x']-(pe['x']-pe['w']/2))/pe['w']
    # (a) vision-language-action model
    panel('aPanel'); lab('aTitle','(a) Vision-language-action model',9,True,'ink')
    tok('aF',seq('f','T')+': unused','gray',True,fill=mix(COL['gray'],.05)); tok('aOut',seq('a','H',predicted=True),'green')
    box('aV','VLM<br>backbone'); box('aA','Action<br>expert','green')
    tok('aObs','<i>f</i><sub>0</sub>'); tok('aText','<i>c</i>','gray'); tok('aN',seq('a','H',noisy=True),'green',True)
    tok('aVisLoss','no future loss','gray',True,fill=mix(COL['gray'],.05)); tok('aActLoss',loss('act'),'green')
    d.edge('aObs','aV',start=(.5,0),end=(frac('aObs','aV'),1)); d.edge('aText','aV','gray',start=(.5,0),end=(frac('aText','aV'),1))
    d.edge('aN','aA','green',start=(.5,0),end=(.5,1)); d.edge('aA','aOut','green',start=(.5,0),end=(.5,1)); d.edge('aV','aA')
    # (b) world-action model
    panel('bPanel'); lab('bTitle','(b) World-action model',9,True,'ink')
    tok('bF',seq('f','T',predicted=True)); tok('bOut',seq('a','H',predicted=True),'green')
    if 'bGT' in spec:   # the recorded future the generative loss compares its output against,
        tok('bGT',seq('f','T'))                                    # drawn like the same tokens in (a) and (c)
        d.edge('bGT','bF',start=(.5,1),end=(.5,0),arrow=False,dash=True)
    box('bV','Generator<br>backbone'); box('bA','Action<br>expert','green')
    tok('bObs','<i>f</i><sub>0</sub>'); tok('bNoisy',seq('f','T',noisy=True),'blue',True); tok('bN',seq('a','H',noisy=True),'green',True)
    tok('bVisLoss',loss('G')+'\u2009: generate the future',size=8.5); tok('bActLoss',loss('act'),'green')
    d.edge('bObs','bV',start=(.5,0),end=(frac('bObs','bV'),1)); d.edge('bNoisy','bV',start=(.5,0),end=(frac('bNoisy','bV'),1))
    d.edge('bV','bF',start=(.5,0),end=(.5,1)); d.edge('bN','bA','green',start=(.5,0),end=(.5,1))
    d.edge('bA','bOut','green',start=(.5,0),end=(.5,1)); d.edge('bV','bA')
    # (c) JAM
    panel('cPanel'); lab('cTitle','(c) JAM: one recipe bridging both families',9,True,'ink')
    narrow = S('cE')['w'] < 2.5          # the two modules sit side by side, so their labels wrap
    box('cE','Frozen target<br>encoder' if narrow else 'Frozen target encoder','orange',
        fill=mix(COL['orange'],.12)); tok('cFuture',seq('f','T'))   # future frames only (the target encoder never sees f_0); blue like every frame token
    box('cH','Prediction<br>head' if narrow else 'Prediction head','orange',
        fill=mix(COL['orange'],.12))
    box('cV','<b>Pretrained backbone</b><br><g><s f=0.77>Generative DiT · SSL · VLM</s></g>',size=9,bold=False)
    tok('cObs','<i>f</i><sub>0</sub>'); tok('cVisLoss',loss('F')+'\u2009: predict the future','orange')
    if 'cLoss' in spec:   # (a) and (b) name their losses only along the bottom, so this is optional
        e=S('cLoss'); d.node('cLoss',u(e['x']),u(e['y']),e['w'],e['h'],loss('F'),color='orange',kind='circle',size=9)
    tok('cOut',seq('a','H',predicted=True),'green'); box('cA','Action<br>expert','green'); tok('cNoise',seq('a','H',noisy=True),'green',True); tok('cActLoss',loss('act'),'green')
    if S('cFuture')['y'] > S('cE')['y']:      # recorded futures sit above the encoder, so aim down
        d.edge('cFuture','cE',start=(.5,1),end=(.5,0))
    else:
        d.edge('cFuture','cE',start=(.5,0),end=(.5,1))
    if 'cG' in spec:            # the target and prediction tokens are drawn, and the loss joins them
        tok('cG',seq('g','T'),'orange'); tok('cP',seq('g','T',predicted=True),'orange',True)
        d.edge('cE','cG','orange'); d.edge('cH','cP','orange')
        d.edge('cG','cLoss','orange',start=(.5,1),end=(.5,0),arrow=False,dash=True)
        d.edge('cLoss','cP','orange',start=(.5,1),end=(.5,0),arrow=False,dash=True)
    else:                       # what the loss compares are the two modules themselves
        side = S('cE')['y'] == S('cH')['y']
        left, right = ('cE','cH') if S('cE')['x'] < S('cH')['x'] else ('cH','cE')
        if 'cLoss' in spec:     # via the named circle
            a,b = (left,'cLoss'),('cLoss',right)
        else:                   # or as one plain link, the loss being named along the bottom
            a,b = (left,right),None
        for pair in (a,b):
            if pair is None: continue
            if side: d.edge(pair[0],pair[1],'orange',start=(1,.5),end=(0,.5),arrow=False,dash=True)
            else:    d.edge(pair[0],pair[1],'orange',start=(.5,0),end=(.5,1),arrow=False,dash=True)
    d.edge('cObs','cV',start=(.5,0),end=(.5,1))
    d.edge('cV','cH','orange',start=(frac('cH','cV'),0),end=(.5,1))   # leaves the backbone under the head
    lab('cHidden','<i>h</i><sup>\u2009(ℓ)</sup>',9,color='orange')
    d.edge('cV','cA'); d.edge('cNoise','cA','green',start=(.5,0),end=(.5,1)); d.edge('cA','cOut','green',start=(.5,0),end=(.5,1))
    return d


def build_method_v2():
    """Figure 2 after the 2026-09-05 re-layout, driven by Figures/fig2_layout.json (TikZ centimetres).

    ``u`` converts centimetres to the diagram units of this generator (1 unit = 1.5 cm)."""
    spec=json.loads((ROOT/'Figures/fig2_layout.json').read_text()); spec.pop('_comment',None)
    u=lambda cm: cm/1.5
    S=lambda k: spec[k]
    P,D=S('trainingPanel'),S('deploymentPanel')
    d=Diagram('Figure 2',u(max(P['x1'],D['x1'])),u(min(P['y0'],D['y0'])),u(max(P['y1'],D['y1'])))
    d.panel('trainingPanel',u(P['x0']),u(P['y0']),u(P['x1']),u(P['y1'])); d.panel('deploymentPanel',u(D['x0']),u(D['y0']),u(D['x1']),u(D['y1']))
    def lab(key,text,size=7.5,bold=False,color='muted'):
        e=S(key); d.label(key,u(e['x']),u(e['y']),e['w'],e['h'],e.get('text',text),size,bold,color)
    def box(key,text,color,size=8,bold=False,fill=None):
        e=S(key); d.node(key,u(e['x']),u(e['y']),e['w'],e['h'],text,color=color,bold=bold,fill=fill if fill else mix(COL[color],.10),size=size)
    def row(key,color,dash=False):
        e=S(key); d.row(key,u(e['x']),u(e['y']),e['count'],color,dash,pitch=u(.4))
    def pair(key,text,color,dash=False,w=.55):
        """A token sequence as two labelled boxes with an ellipsis, rather than blank squares with
        the sequence written beside them, on a band that holds the two together."""
        e=S(key); a,b=text.split(' \u00b7\u00b7\u00b7 ')
        d.pair(key,u(e['x']),u(e['y']),a,b,color,dash,w=e.get('boxw',w),gap=e.get('gap',.46),
               band=e.get('band',.08))
    def single(key,text,color,dash=False):
        """A lone token on the same kind of band a pair sits on, so the two read as one row."""
        e=S(key); w=e.get('boxw',.75); band=e.get('band',.08)
        if not band:
            d.node(key,u(e['x']),u(e['y']),w,.42,text,color=color,kind='token',dash=dash,size=7); return
        pad,vpad=band,band*0.375
        d.node(key,u(e['x']),u(e['y']),w+2*pad,.42+2*vpad,color=color,kind='band',fill=mix(COL[color],.05))
        child=d.node(key+'_0',u(e['x']),u(e['y']),w,.42,text,color=color,kind='token',dash=dash,size=7)
        d.nodes[child]['parent']=key
    def compare(a,b,color='orange'):
        """Join two shapes a loss compares, along whichever axis they are stacked on and from
        whichever side the second one lies, so the link never crosses either of them."""
        ea,eb=S(a),S(b)
        if abs(ea['y']-eb['y']) >= abs(ea['x']-eb['x']):        # one above the other
            lo,hi=(a,b) if ea['y']>eb['y'] else (b,a)
            d.edge(lo,hi,color,start=(.5,1),end=(.5,0),arrow=False,dash=True)
        else:                                                   # side by side
            lf,rt=(a,b) if ea['x']<eb['x'] else (b,a)
            d.edge(lf,rt,color,start=(1,.5),end=(0,.5),arrow=False,dash=True)
    def frac(key,parent):   # where `key` sits along the top or bottom edge of `parent`
        e,pe=S(key),S(parent); return (e['x']-(pe['x']-pe['w']/2))/pe['w']
    def img(key,image,color='blue',dx=0.0,base=None):
        e=S(base or key); d.node(key,u(e['x']+dx),u(e['y']+dx),e['w'],e['h'],color=color,kind='image',image=image)
    lab('trainingTitle','Training',8,True,'ink'); lab('deploymentTitle','Deployment',8,True,'ink')
    lab('instruction','<i>“Put the bowl on the plate”</i>',7.5,color='ink')
    # an explicit break: the label is wider than the trapezoid, and LibreOffice wraps it while
    # PowerPoint honours word_wrap=False and lets it overrun the shape and the snowflake
    for key,label in [('text','Text<br>encoder'),('input','Input<br>encoder')]:
        e=S(key); d.node(key,u(e['x']),u(e['y']),e['w'],e['h'],label,color='gray',kind='encoder',bold=True,fill=mix(COL['gray'],.12),size=8)
        d.label(key+'Snow',u(e['x']+e['w']/2+0.29-0.32),u(e['y']+e['h']/2-0.18),.28,.3,'❄',8,color='blue'); d.nodes[key+'Snow']['parent']=key
    img('observation','spatial_current.png'); lab('obsLabel','<i>x</i><sub>t</sub>',7.5,color='blue')
    if S('observation')['x'] < S('input')['x']:   # the frame feeds the encoder from its left
        d.edge('observation','input',start=(1,.5),end=(0,.5))
    else:
        d.edge('observation','input',start=(0,.5),end=(1,.5))
    tokens_drawn = 'textTokens' in spec
    if tokens_drawn:   # the encoders may hand their tokens to the backbone through a drawn row
        row('textTokens','gray'); row('visualTokens','blue'); lab('f0Label','<i>f</i><sub>0</sub>',7.5,color='blue')
        d.edge('text','textTokens','gray',start=(.5,1),end=(.5,0))
        d.edge('input','visualTokens',start=(.5,1),end=(.5,0))
    # a narrow backbone breaks its name and its list, the same way the deployment panel's does
    box('backbone',
        '<b>Pretrained<br>backbone</b><br><g><s f=0.79>Generative DiT · SSL · VLM</s></g>'
        if S('backbone')['w']<3.2 else
        '<b>Pretrained backbone</b><br><g><s f=0.79>Generative DiT · SSL · VLM</s></g>','blue',8)
    box('action','Action expert','green',8,True)
    bb=S('backbone'); ac=S('action')
    for src,dst,col in ((('textTokens' if tokens_drawn else 'text'),'backbone','gray'),
                       (('visualTokens' if tokens_drawn else 'input'),'backbone','blue')):
        d.edge(src,dst,col,start=(.5,1),end=((S(src)['x']-(bb['x']-bb['w']/2))/bb['w'],0))
    d.edge('backbone','action'); lab('kv','<b>KV</b>',7.5,color='blue')
    single('state','<i>s</i><sub>0</sub>','green')
    pair('noisyActions','<i>ã</i><sub>1</sub> ··· <i>ã</i><sub>H</sub>','green',True,w=.72)
    for key in ('state','noisyActions'):
        d.edge(key,'action','green',start=(.5,1),end=((S(key)['x']-(ac['x']-ac['w']/2))/ac['w'],0))
    pair('actionTokens','<i>â</i><sub>1</sub> ··· <i>â</i><sub>H</sub>','green',w=.72)
    d.edge('action','actionTokens','green',start=((S('actionTokens')['x']-(ac['x']-ac['w']/2))/ac['w'],1),end=(.5,0))
    e=S('actionLoss'); d.node('actionLoss',u(e['x']),u(e['y']),e['w'],e['h'],loss('act'),color='green',kind='circle',size=7.5)
    compare('actionTokens','actionLoss','green')
    box('head','Prediction<br>head <i>p</i>','orange',8,True,mix(COL['orange'],.12))
    if S('head')['y'] < S('backbone')['y']-0.4:      # the head hangs below the backbone
        d.edge('backbone','head','orange',start=(frac('head','backbone'),1),end=(.5,0))
    else:
        d.edge('backbone','head','orange',start=(0,.5),end=(1,.5))
    lab('hidden','<i>h</i><sup>\u2009(ℓ)</sup>',8.5,color='orange')
    tokens = 'predTokens' in spec      # the branch may name its tokens, or just link head to encoder
    if tokens:
        pair('predTokens',seq('g','T',predicted=True),'orange',True)
        pair('targetTokens',seq('g','T'),'orange')
    if tokens: d.edge('head','predTokens','orange',start=(.5,1),end=(.5,0))
    e=S('foresightLoss'); d.node('foresightLoss',u(e['x']),u(e['y']),e['w'],e['h'],loss('F'),color='orange',kind='circle',size=7.5)
    if 'goLabel' in spec: lab('goLabel','<i>g</i><sub>k</sub>',8.5,color='orange')   # the target, named on its link to the loss
    if 'cacheBox' in spec:   # optional: enclose the branch the caption calls cacheable
        e=S('cacheBox')
        d.node('cacheBox',u(e['x']),u(e['y']),e['w'],e['h'],color='orange',kind='cache',
               dash=True,fill='none')
    if 'cacheNote' in spec:  # optional, and independent of the enclosure
        lab('cacheNote','<i>computed once, then cached</i>',7.5,color='orange')
    if tokens:
        compare('predTokens','foresightLoss'); compare('foresightLoss','targetTokens')
    else:                               # what the loss compares are the two modules themselves
        compare('head','foresightLoss'); compare('foresightLoss','teacher')
    e=S('teacher')
    # a narrow encoder breaks its name and its list of backbones, the way the module boxes do
    # a trapezoid narrows toward its base, so a narrow one takes a shorter list on a single line
    # the snowflake already says it is frozen
    name_t=('<b>Target encoder <i>E</i><sub>φ</sub></b>' if e['w']>=3.0 else '<b>Target<br>encoder <i>E</i><sub>φ</sub></b>')
    list_t=('Generative DiT · SSL · VLM · VAE' if e['w']>=3.0 else 'DiT · SSL · VLM · VAE')
    lab_t=name_t+'<br><g><s f=0.79>'+list_t+'</s></g>'
    d.node('teacher',u(e['x']),u(e['y']),e['w'],e['h'],lab_t,color='orange',kind=e.get('kind','encoder'),size=8,fill=mix(COL['orange'],.12))
    d.label('teacherSnow',u(e['x']+e['w']/2-0.24),u(e['y']+e['h']/2-0.18),.28,.3,'❄',8,color='blue'); d.nodes['teacherSnow']['parent']='teacher'
    if tokens: d.edge('teacher','targetTokens','orange',start=(0,.5),end=(1,.5))
    img('futureBack','spatial_recorded_t8.png','orange',0.06,'future'); img('futureMiddle','spatial_recorded_t8.png','orange',0.03,'future'); img('future','spatial_recorded_t8.png','orange',0.0)
    lab('futureLabel','<i>x</i><sub>t+1:T</sub>',7.5,color='orange')
    fy,ty=S('future')['y'],S('teacher')['y']
    if fy > ty+0.5:                                  # the frames sit above the encoder, straight down
        t=S('teacher'); d.edge('future','teacher','orange',start=(.5,1),end=((S('future')['x']-(t['x']-t['w']/2))/t['w'],0))
    elif S('future')['x'] < S('teacher')['x']:       # or feed it from its left
        d.edge('future','teacher','orange',start=(1,.5),end=(0,.5))
    else:
        d.edge('future','teacher','orange',start=(0,.5),end=(1,.5))
    for key,parent in [('backboneFire','backbone'),('actionFire','action'),('headFire','head')]:
        # top-right corner: the flame rises about 0.32 above its base, so the base sits 0.40 below the top edge
        e=S(parent); d.node(key,u(e['x']+e['w']/2-0.21),u(e['y']+e['h']/2-0.40),.20,.34,color='orange',kind='flame',fill=COL['orange']); d.nodes[key]['parent']=parent
    if 'deployInstruction' in spec:   # the panel need not repeat the training panel's sentence
        lab('deployInstruction','<i>“Put the bowl on the plate”</i>',7.5,color='ink')
    img('deployObs','spatial_current.png')
    if 'deployObsLabel' in spec: lab('deployObsLabel','<i>f</i><sub>0</sub>',7.5,color='blue')   # named beside the frame
    # a narrow deployment backbone breaks its name and its list, so it can sit beside the frame
    box('deployBackbone',
        '<b>Pretrained<br>backbone</b><br><g><s f=0.79>DiT · SSL · VLM</s></g>'
        if S('deployBackbone')['w']<2.8 else
        '<b>Pretrained backbone</b><br><g><s f=0.79>Generative DiT · SSL · VLM</s></g>','blue',8)
    box('deployAction','Action expert','green',8,True)
    pair('deployActions','<i>â</i><sub>1</sub> ··· <i>â</i><sub>H</sub>','green',w=.72)
    if S('deployObs')['x'] < S('deployBackbone')['x']-0.5:   # the frame feeds it from the side
        d.edge('deployObs','deployBackbone',start=(1,.5),end=(0,.5))
    else:
        d.edge('deployObs','deployBackbone',start=(.5,1),end=(.5,0))
    d.edge('deployBackbone','deployAction',start=(.5,1),end=(.5,0))
    lab('deployKV','<b>KV</b>',7.5,color='blue')
    d.edge('deployAction','deployActions','green',start=(.5,1),end=(.5,0))
    return d

def mx_model(d):
    model=ET.Element('mxGraphModel',dx='1200',dy='700',grid='1',gridSize='10',guides='1',
        tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',
        pageWidth=str(round(d.width)),pageHeight=str(round(d.height)),math='0',shadow='0')
    root=ET.SubElement(model,'root'); ET.SubElement(root,'mxCell',id='0'); ET.SubElement(root,'mxCell',id='1',parent='0')
    for key,n in d.nodes.items():
        shape={'encoder':'shape=trapezoid;direction=south;', 'circle':'ellipse;',
               'text':'text;strokeColor=none;fillColor=none;', 'group':'group;',
               'flame':'shape=mxgraph.basic.drop;direction=north;',
               'image':'shape=image;aspect=fixed;imageAspect=1;'}
        st=shape.get(n['kind'],'rounded=1;absoluteArcSize=1;arcSize=6;')
        st+=f"html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily=Times New Roman;fontSize={n['size']:.3f};fontColor=#{COL['ink']};fontStyle={int(n['bold'])};spacing=0;spacingLeft=4;spacingRight=4;"
        if n['label_shift']:
            st+=f"spacingLeft=0;spacingRight={-2*n['label_shift']:.3f};"
        if '❄' in n['label']: st+='fontFamily=DejaVu Sans;'
        if n['kind'] not in ('text','group'):
            st+=f"strokeColor=#{n['stroke']};fillColor=#{n['fill']};strokeWidth=1.6;"
        if n['kind']=='text': st+=f"fontColor=#{n['stroke']};"
        if n['dash']: st+='dashed=1;dashPattern=4 3;'
        if n['image']:
            raw=(ROOT/'Figures/foresight_decode'/n['image']).read_bytes()
            st+='image=data:image/png,'+base64.b64encode(raw).decode()+';'
        parent=n.get('parent','1')
        x,y,w,h=n['x'],n['y'],n['w'],n['h']
        if n['kind']=='encoder':
            ext=(0.18 if h<=0.6*CM else 0.29)*CM
            st+=f'fixedSize=1;size={2*ext:.3f};'
            x-=ext; w+=2*ext
        cell=ET.SubElement(root,'mxCell',id=key,value=n['label'],style=st,vertex='1',parent=parent)
        if parent!='1':
            pn=d.nodes[parent]; pext=((0.18 if pn['h']<=0.6*CM else 0.29)*CM) if pn['kind']=='encoder' else 0
            x-=pn['x']-pext; y-=pn['y']
        ET.SubElement(cell,'mxGeometry',x=f'{x:.3f}',y=f'{y:.3f}',width=f"{w:.3f}",height=f"{h:.3f}",attrib={'as':'geometry'})
    for e in d.edges:
        st=f"edgeStyle=none;rounded=0;html=1;strokeColor=#{e['color']};strokeWidth=1.8;endArrow={'block' if e['arrow'] else 'none'};endFill=1;"
        st+=f"exitX={e['start'][0]};exitY={e['start'][1]};exitDx=0;exitDy=0;exitPerimeter=0;entryX={e['end'][0]};entryY={e['end'][1]};entryDx=0;entryDy=0;entryPerimeter=0;"
        if e['dash']: st+='dashed=1;dashPattern=4 3;'
        cell=ET.SubElement(root,'mxCell',id=e['id'],style=st,edge='1',parent='1',source=e['source'],target=e['target'])
        ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
    return model


def svg_text(n):
    lines=RichText(n['label']).lines
    fs=n['size']; lh=fs*1.12
    baseline=n['y']+n['h']/2-(len(lines)-1)*lh/2+fs*.30
    result=[]
    for i,line in enumerate(lines):
        parts=[]; measured=[]
        surface=cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1)
        ctx=cairo.Context(surface)
        for value,bold,italic,script,muted,small in line:
            family='DejaVu Sans' if '❄' in value else 'Liberation Serif'
            size=fs*(.72 if script else 1)*small
            ctx.select_font_face(family,cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD if bold or n['bold'] else cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(size)
            measured.append((value,bold,italic,script,family,size,
                             ctx.text_extents(value.replace('̂','').replace('̃',''))[4],muted))
        cursor=n['x']+(n['w']-sum(r[-1] for r in measured))/2+n['label_shift']
        for value,bold,italic,script,family,size,advance,muted in measured:
            shift=fs*(.22 if script==1 else -.40 if script==-1 else 0)
            color=COL['muted'] if muted else (n['stroke'] if n['kind'] in ('text','circle') else COL['ink'])
            plain=value.replace('̂','').replace('̃','')
            parts.append(f'<text x="{cursor:.3f}" y="{baseline+i*lh+shift:.3f}" font-family="{family}" font-size="{size:.3f}" fill="#{color}" font-weight="{"bold" if bold or n["bold"] else "normal"}" font-style="{"italic" if italic else "normal"}">{html.escape(plain)}</text>')
            ay=baseline+i*lh+shift-size*.80
            if '̂' in value:
                parts.append(f'<path d="M{cursor+size*.10},{ay} l{size*.20},{-size*.16} l{size*.20},{size*.16}" fill="none" stroke="#{color}" stroke-width="1"/>')
            if '̃' in value:
                parts.append(f'<path d="M{cursor+size*.08},{ay} q{size*.10},{-size*.15} {size*.24},0 t{size*.24},0" fill="none" stroke="#{color}" stroke-width="1"/>')
            cursor+=advance
        color=n['stroke'] if n['kind']=='text' else COL['ink']
        result.append(''.join(parts))
    return ''.join(result)


def svg_preview(d, mxfile):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{d.width}" height="{d.height}" viewBox="0 0 {d.width} {d.height}" content="{html.escape(mxfile,quote=True)}">',
      '<defs>'+''.join(f'<marker id="arr_{col}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0 0 L8 3.5 L0 7 Z" fill="#{col}"/></marker>' for col in set(e['color'] for e in d.edges))+'</defs>',
      f'<rect width="{d.width}" height="{d.height}" fill="white"/>']
    for n in d.nodes.values():
        if n['kind']=='panel':
            parts.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="7" stroke="#{n["stroke"]}" fill="#{n["fill"]}" stroke-width="1.5"/>')
    for e in d.edges:
        x1,y1=d.point(e['source'],e['start']); x2,y2=d.point(e['target'],e['end'])
        marker=f' marker-end="url(#arr_{e["color"]})"' if e['arrow'] else ''
        dash=' stroke-dasharray="5 4"' if e['dash'] else ''
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#{e["color"]}" stroke-width="1.8"{marker}{dash}/>')
    for n in d.nodes.values():
        kind=n['kind']; x,y,w,h=(n[k] for k in ('x','y','w','h'))
        attrs=f'fill="#{n["fill"]}" stroke="#{n["stroke"]}" stroke-width="1.6"'
        if n['dash']: attrs+=' stroke-dasharray="5 4"'
        if kind in ('panel','group'): continue
        if kind=='cache':
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0.10*min(w,h)}" fill="none" stroke="#{n[chr(34)+chr(34)] if False else n["stroke"]}" stroke-width="1.0" stroke-dasharray="5 4"/>')
            continue
        if kind=='band':
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0.14*min(w,h)}" fill="#{n["fill"]}" stroke="none"/>')
            continue
        if kind=='image':
            raw=base64.b64encode((ROOT/'Figures/foresight_decode'/n['image']).read_bytes()).decode()
            parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" xlink:href="data:image/png;base64,{raw}"/>')
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#{n["stroke"]}" stroke-width="1.6"/>')
        elif kind=='encoder':
            inset=(0.18 if h<=0.6*CM else 0.29)*CM
            parts.append(f'<polygon points="{x-inset},{y} {x+w+inset},{y} {x+w-inset},{y+h} {x+inset},{y+h}" {attrs}/>')
        elif kind=='circle': parts.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" {attrs}/>')
        elif kind=='flame': parts.append(f'<path d="M{x+w/2},{y} Q{x+1.6*w},{y+h*.8} {x+w/2},{y+h} Q{x-w*.6},{y+h*.8} {x+w/2},{y}" {attrs}/>')
        elif kind!='text': parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" {attrs}/>')
        if n['label']: parts.append(svg_text(n))
    return ''.join(parts)+'</svg>'


RECT_SITES={(0.5,0):0,(0,0.5):1,(0.5,1):2,(1,0.5):3}
ELLIPSE_SITES={(0.5,0):0,(0,0.5):2,(0.5,1):4,(1,0.5):6}
TRAP180_SITES={(0.5,0):2,(0,0.5):3,(0.5,1):0,(1,0.5):1}   # trapezoid drawn rotated by 180 degrees
FLAME_OUTER=[((0,0),(-0.6,0.5),(-0.42,1.05),(0,1.45)),((0,1.45),(0.42,1.05),(0.6,0.5),(0,0))]
FLAME_INNER=[((0,0.1),(-0.26,0.38),(-0.18,0.68),(0,0.88)),((0,0.88),(0.18,0.68),(0.26,0.38),(0,0.1))]
FLAME_SCALE=0.19  # TikZ flame scale (units of cm)


def bezier_points(curves,samples=14):
    pts=[]
    for p0,p1,p2,p3 in curves:
        for k in range(samples):
            t=k/samples; u=1-t
            pts.append((u*u*u*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t*t*t*p3[0],
                        u*u*u*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t*t*t*p3[1]))
    return pts


def add_flame(slide,cx_px,base_px,name,curves,color):
    """Freeform flame with its base at (cx,base) px, extending upward; TikZ path sampled as a polygon."""
    s=FLAME_SCALE*CM
    pts=[(Inches((cx_px+x*s)/96),Inches((base_px-y*s)/96)) for x,y in bezier_points(curves)]
    fb=slide.shapes.build_freeform(pts[0][0],pts[0][1],scale=1.0)
    fb.add_line_segments(pts[1:],close=True)
    shape=fb.convert_to_shape()
    shape.name=name; shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor.from_string(color); shape.line.fill.background()
    return shape


def run_font(value,bold,italic,script):
    """Times for text, a math font for single-letter variables, subscripts and symbols (as in the paper).

    Every name here is a font PowerPoint carries, so the sources open anywhere without installing
    anything. The exporter maps them to what the paper prints: Cambria Math becomes Latin Modern
    Math, and Segoe UI Symbol becomes DejaVu Sans, which the render container has and Segoe is not.
    """
    if '❄' in value: return 'Segoe UI Symbol'
    if 'ℒ' in value or 'ℓ' in value: return 'Cambria Math'
    if script and value not in ('act',): return 'Cambria Math'
    if italic and len(value.strip())<=2: return 'Cambria Math'
    return 'Times New Roman'


def pptx_export(diagrams,path):
    prs=Presentation(); prs.slide_width=Inches(max(d.width for d in diagrams)/96)
    prs.slide_height=Inches(max(d.height for d in diagrams)/96)
    for d in diagrams:
        slide=prs.slides.add_slide(prs.slide_layouts[6]); objects={}; extra={}
        for key,n in d.nodes.items():
            x,y,w,h=(Inches(n[k]/96) for k in ('x','y','w','h'))
            if n['kind']=='image':
                shape=slide.shapes.add_picture(str(ROOT/'Figures/foresight_decode'/n['image']),x,y,w,h)
                # the frames carry the colour of the path they feed, the way the SVG mirror draws them
                shape.line.color.rgb=RGBColor.from_string(n['stroke']); shape.line.width=Pt(1.2)
            elif n['kind'] in ('text','group'):
                shape=slide.shapes.add_textbox(x,y,w,h)
            elif n['kind']=='flame':
                cx=n['x']+n['w']/2; base=n['y']+n['h']/2
                shape=add_flame(slide,cx,base,key,FLAME_OUTER,COL['orange'])
                inner=add_flame(slide,cx,base,key+'_inner',FLAME_INNER,'F5C542')
                extra.setdefault(n.get('parent'),[]).append(inner)
            else:
                typ={'circle':MSO_SHAPE.OVAL,'encoder':MSO_SHAPE.TRAPEZOID,
                     'cache':MSO_SHAPE.ROUNDED_RECTANGLE}.get(n['kind'],MSO_SHAPE.ROUNDED_RECTANGLE)
                if n['kind']=='encoder':
                    ext=Inches(((0.18 if n['h']<=0.6*CM else 0.29)*CM)/96)
                    shape=slide.shapes.add_shape(typ,x-ext,y,w+2*ext,h)
                    shape.rotation=180; shape.adjustments[0]=.5
                else:
                    shape=slide.shapes.add_shape(typ,x,y,w,h)
                if n['fill']=='none':
                    shape.fill.background()           # an outline over whatever it encloses
                else:
                    shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor.from_string(n['fill'])
                if n['kind']=='band':
                    shape.line.fill.background()      # colour alone holds the pair together
                elif n['kind']=='cache':
                    shape.line.color.rgb=RGBColor.from_string(n['stroke']); shape.line.width=Pt(0.8)
                else:
                    shape.line.color.rgb=RGBColor.from_string(n['stroke']); shape.line.width=Pt(1.2)
                if n['dash']: shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH
                if typ==MSO_SHAPE.ROUNDED_RECTANGLE:
                    # module boxes get a fixed 0.15 cm corner radius so short and tall boxes look equally round;
                    # the PowerPoint adjustment is the radius as a fraction of the shorter side
                    shape.adjustments[0]=min(.5,BOX_RADIUS*CM/min(n['w'],n['h'])) if n['kind'] in ('box','band') else {'panel':.06,'token':.16}.get(n['kind'],.06)   # a band's corners are as round as a module box's
            try: shape.shadow.inherit=False    # the diagram is flat; PowerPoint's default drop
            except Exception: pass             # shadow only muddies the thin strokes
            shape.name=key; objects[key]=shape
            if n['label']:
                target=slide.shapes.add_textbox(x+Inches(n['label_shift']/96),y-Inches(0.04*CM/96),w,h) if n['kind']=='encoder' else shape
                if target is not shape: target.name=key+'_label'; objects[key+'_label']=target
                tf=target.text_frame; tf.clear(); tf.word_wrap=False
                tf.margin_left=tf.margin_right=Pt(1); tf.margin_top=tf.margin_bottom=Pt(0)
                tf.vertical_anchor=MSO_ANCHOR.MIDDLE
                for i,line in enumerate(RichText(n['label']).lines):
                    p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
                    p.alignment=PP_ALIGN.CENTER; p.space_before=p.space_after=Pt(0); p.line_spacing=1.0
                    for value,bold,italic,script,muted,small in line:
                        # sub/superscripts: letters italic, digits and punctuation upright (LaTeX convention)
                        if not script or value in ('act',) or 'ℓ' in value:
                            pieces=[(value,italic or (bool(script) and value not in ('act',) and 'ℓ' not in value))]
                        else:
                            pieces=[(m.group(0),bool(re.match(r'[A-Za-zα-ωℓφ]',m.group(0)))) for m in re.finditer(r'[A-Za-zα-ωℓφ]+|[^A-Za-zα-ωℓφ]+',value)]
                        for text,ital in pieces:
                            run=p.add_run(); run.text=text; run.font.name=run_font(text,bold,ital,script)
                            run.font.size=Pt(n['size']*.75*(.88 if script else 1)*small)
                            run.font.bold=bold or n['bold']; run.font.italic=ital
                            run.font.color.rgb=RGBColor.from_string(COL['muted'] if muted else (n['stroke'] if n['kind'] in ('text','circle') else COL['ink']))
                            if script: run._r.get_or_add_rPr().set('baseline','-25000' if script==1 else '30000')
        prim=dict(objects)  # primary shapes before grouping
        # invisible anchors for connector ends that do not sit on a standard connection site
        def site_for(key,anchor):
            n=d.nodes[key]; kind=n['kind']
            if kind=='group':
                cnt=sum(1 for c in d.nodes if c.startswith(key+'_') and d.nodes[c].get('parent')==key)
                idx={(0,0.5):0,(1,0.5):cnt-1}.get(anchor,cnt//2)
                return prim[f'{key}_{idx}'],RECT_SITES.get(anchor)
            table={'circle':ELLIPSE_SITES,'encoder':TRAP180_SITES}.get(kind,RECT_SITES)
            return prim[key],table.get(anchor)
        for e in d.edges:
            for end,key,anchor in (('start',e['source'],e['start']),('end',e['target'],e['end'])):
                shape,idx=site_for(key,anchor)
                if idx is None:
                    px,py=d.point(key,anchor)
                    # the line glues to the anchor's site 0, its top centre, so that point is the anchor's top edge
                    a=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches((px-0.5)/96),Inches(py/96),Inches(1/96),Inches(1/96))
                    a.fill.background(); a.line.fill.background(); a.name=f"{e['id']}_{end}_anchor"
                    extra.setdefault(key,[]).append(a); e[end+'_site']=(a,0)
                else:
                    e[end+'_site']=(shape,idx)
        for key,n in d.nodes.items():
            children=[objects[child] for child,c in d.nodes.items() if c.get('parent')==key]
            children+=extra.get(key,[])
            if key+'_label' in objects: children.insert(0,objects[key+'_label'])
            if children:
                group=slide.shapes.add_group_shape([objects[key],*children])
                group.name=key+'_editable_group'
                objects[key]=group
        for e in d.edges:
            x1,y1=d.point(e['source'],e['start']); x2,y2=d.point(e['target'],e['end'])
            c=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,*[Inches(v/96) for v in (x1,y1,x2,y2)])
            c.name=e['id']; c.line.color.rgb=RGBColor.from_string(e['color']); c.line.width=Pt(1.35)
            # attach both ends to shape connection sites without moving the line (python-pptx only
            # knows four sites; ellipses and rotated trapezoids need other indices)
            s0,i0=e['start_site']; s1,i1=e['end_site']
            cnv=c._element.nvCxnSpPr.cNvCxnSpPr
            cnv._add_stCxn(id=s0.shape_id,idx=i0); cnv._add_endCxn(id=s1.shape_id,idx=i1)
            if e['dash']:
                # A comparison link spans a gap of well under a centimetre, where the coarse
                # DASH used for box outlines fits one dash and a stub and reads as lopsided.
                # Finer than the 1.2 pt DASH of a dashed box outline, so a loss link reads as a
                # different kind of relation than the flow arrows and does not look heavy.
                c.line.dash_style=MSO_LINE_DASH_STYLE.SQUARE_DOT
                c.line.width=Pt(0.9)
            if e['arrow']:
                end=OxmlElement('a:tailEnd'); end.set('type','triangle'); c.line._get_or_add_ln().append(end)
        slide.notes_slide.notes_text_frame.text='Editable source of '+d.name+'.'
    prs.save(path)
    # every shape, connector and stroke: no theme shadow (python-pptx's theme puts one on all three effect styles)
    import subprocess; subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),'strip_pptx_effects.py'), path], check=True, stdout=subprocess.DEVNULL)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--figure2-relayout',action='store_true',
                        help='write only the re-laid-out Figure 2 (2026-09-05 layout) under distinct file names')
    parser.add_argument('--figure1-relayout',action='store_true',
                        help='write only Figure 1 from Figures/fig1_layout.json under distinct file names')
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    if args.figure2_relayout:
        return main_figure2_relayout(out)
    if args.figure1_relayout:
        return main_single(out,'Figure_1_relayout',build_designspace_v2,'Figures/fig_designspace.tex via fig1_layout.json')
    names=['JAM_Figures_1_2.drawio','JAM_Figures_1_2.pptx','Figure_1.drawio','Figure_2.drawio',
           'Figure_1.drawio.svg','Figure_2.drawio.svg','Figure_1.png','Figure_2.png',
           'Figure_1.pdf','Figure_2.pdf','editable_manifest.json']
    assert not any((out/n).exists() for n in names), 'Refusing to overwrite editable copies. Choose a new output directory.'
    diagrams=[build_designspace(),build_method()]
    combined=ET.Element('mxfile',host='app.diagrams.net',agent='JAM editable export',version='24.7.17',type='device')
    for i,d in enumerate(diagrams,1):
        model=mx_model(d)
        page=ET.SubElement(combined,'diagram',id=f'figure_{i}',name=d.name); page.append(model)
        single=ET.Element('mxfile',host='app.diagrams.net',type='device')
        single.append(ET.fromstring(ET.tostring(page)))
        mxtext=ET.tostring(single,encoding='unicode')
        (out/f'Figure_{i}.drawio').write_text(mxtext)
        svg=svg_preview(d,mxtext)
        (out/f'Figure_{i}.drawio.svg').write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(out/f'Figure_{i}.png'),scale=1.5)
        cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(out/f'Figure_{i}.pdf'))
    (out/'JAM_Figures_1_2.drawio').write_text(ET.tostring(combined,encoding='unicode'))
    pptx_export(diagrams,out/'JAM_Figures_1_2.pptx')
    report={'auto_sync':False,'diagrams':[
        dict(name=d.name,nodes=len(d.nodes),edges=len(d.edges),width=d.width,height=d.height) for d in diagrams]}
    (out/'editable_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main_figure2_relayout(out):
    return main_single(out,'Figure_2_relayout',build_method_v2,'Figures/fig_method.tex (2026-09-05 re-layout)')


def main_single(out,stem,builder,source):
    names=[f'{stem}.drawio',f'{stem}.drawio.svg',f'{stem}.png',f'{stem}.pdf',f'{stem}.pptx',f'{stem}_manifest.json']
    assert not any((out/n).exists() for n in names), 'Refusing to overwrite editable copies. Choose a new output directory.'
    d=builder()
    single=ET.Element('mxfile',host='app.diagrams.net',agent='JAM editable export',version='24.7.17',type='device')
    page=ET.SubElement(single,'diagram',id=stem.lower(),name=d.name); page.append(mx_model(d))
    mxtext=ET.tostring(single,encoding='unicode')
    (out/f'{stem}.drawio').write_text(mxtext)
    svg=svg_preview(d,mxtext)
    (out/f'{stem}.drawio.svg').write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(out/f'{stem}.png'),scale=1.5)
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(out/f'{stem}.pdf'))
    pptx_export([d],out/f'{stem}.pptx')
    report={'source':source,'auto_sync':False,
            'diagram':dict(name=d.name,nodes=len(d.nodes),edges=len(d.edges),width=d.width,height=d.height)}
    (out/f'{stem}_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__': main()

"""Compact two-column method figure with separate reasoning and memory.

Usage: python draw_framework_v3.py (portable; uses bundled assets only)
SVG keeps editable text; PDF embeds fonts; PNG is exported at 600 dpi.
All photographs are unmodified experiment artifacts. The semantic-mask tile is
a categorical scientific visualization of archived masks. Surface glyphs are
schematic, not a predicted normal image.
"""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.path import Path as MPath
import matplotlib.patheffects as pe

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)

P = dict(ink='#29322E', muted='#5F6C64', edge='#A9B5A7',
         green_bg='#F0F5EE', green_head='#DEE8D9', green='#607D59',
         green_card='#E1EAD9', blue='#AAC4D1', blue_deep='#42677D',
         blue_light='#E8F0F4', cream='#FAF6ED', cream_head='#EEE3CE',
         ochre='#917A43', red='#AC5146', white='#FFFFFF',
         agent_head='#55798B', memory_head='#627E59', support='#F7F8F5')
W, H = 1800, 840
WIDTH_IN = 180 / 25.4  # exact 180 mm, full two-column width
SCALE = WIDTH_IN * 72 / W
F_MAIN, F_SUB = 40 * SCALE, 25 * SCALE
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                     'svg.fonttype': 'none', 'image.composite_image': False, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.unicode_minus': False})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * H / W), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(H, 0))
ax.set_axis_off()
texts = []


def box(x, y, w, h, fill='white', edge=None, lw=1.0, radius=0, z=2):
    if radius:
        patch = FancyBboxPatch((x,y),w,h,
            boxstyle=f'round,pad=0,rounding_size={radius}',
            fc=fill, ec=edge or 'none', lw=lw*SCALE, zorder=z)
    else:
        patch = Rectangle((x,y),w,h,fc=fill,ec=edge or 'none',lw=lw*SCALE,zorder=z)
    ax.add_patch(patch)
    return patch


def txt(x,y,s,main=False,bold=False,italic=False,color=None,ha='center',
        va='center',z=8,bg=None):
    t=ax.text(x,y,s,fontsize=F_MAIN if main else F_SUB,
              fontweight='bold' if bold else 'normal',
              fontstyle='italic' if italic else 'normal',
              color=color or P['ink'],ha=ha,va=va,linespacing=1.18,zorder=z,
              bbox=dict(facecolor=bg,edgecolor='none',pad=1.3) if bg else None)
    texts.append(t)
    return t


def line(points,color=None,lw=2.1,dash=None,z=4,halo=False):
    xs,ys=zip(*points)
    a,=ax.plot(xs,ys,color=color or P['ink'],lw=lw*SCALE,
              linestyle=dash or '-',solid_capstyle='butt',zorder=z)
    if halo:a.set_path_effects([pe.Stroke(linewidth=(lw+8)*SCALE,foreground='white'),pe.Normal()])
    return a


def arrow(points,color=None,lw=2.4,dash=None,z=5,halo=False):
    color=color or P['ink']
    codes=[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)
    a=FancyArrowPatch(path=MPath(points,codes),arrowstyle='-|>',
                      mutation_scale=15*SCALE,lw=lw*SCALE,color=color,
                      linestyle=dash or '-',zorder=z,joinstyle='miter')
    if halo:a.set_path_effects([pe.Stroke(linewidth=(lw+8)*SCALE,foreground='white'),pe.Normal()])
    ax.add_patch(a)
    return a


def panel(x,y,w,h,title,kind='green'):
    bg=P['green_bg'] if kind=='green' else P['cream']
    head=P['green_head'] if kind=='green' else P['cream_head']
    box(x,y,w,h,bg,z=0)
    box(x,y,w,55,head,z=1)
    txt(x+20,y+28,title,main=True,bold=True,italic=True,
        color=P['green'] if kind=='green' else P['ochre'],ha='left')


def photo(name,x,y,w,h,border=True):
    im=Image.open(ASSETS/name).convert('RGB')
    iw,ih=im.size
    scale=min(w/iw,h/ih)
    aw,ah=iw*scale,ih*scale
    xx,yy=x+(w-aw)/2,y+(h-ah)/2
    ax.imshow(im,extent=(xx,xx+aw,yy+ah,yy),zorder=3,interpolation='lanczos')
    if border:box(xx,yy,aw,ah,'none',P['edge'],lw=1.3,z=4)


def dot(x,y,color):
    ax.plot([x],[y],'o',color=color,markersize=4*SCALE,zorder=7)


provenance=json.loads((HERE/'asset_provenance.json').read_text())
examples={r['name']:r for r in provenance['examples']}
for filename in ['reference.png','depth.png','albedo.png','shading.png',
                 'environment.png','masks.png','render.png','initial.png','best.png']:
    if not (ASSETS/filename).is_file():
        raise FileNotFoundError(ASSETS/filename)


def core(x,y,w,h,title,memory=False):
    """Only the three principal modules get a large title and dark band."""
    edge=P['memory_head'] if memory else P['agent_head']
    fill=P['green_bg'] if memory else P['blue_light']
    box(x,y,w,h,fill,edge,lw=2.3,radius=5)
    box(x,y,w,57,edge)
    txt(x+w/2,y+29,title,main=True,bold=True,color='white')


# Secondary evidence is condensed into a narrow, neutral supporting column.
box(10,10,270,700,P['support'],z=0)
txt(145,38,'Image & fixed evidence',bold=True,color=P['muted'])
txt(145,77,r'Input $I$')
photo('reference.png',40,99,210,157)
arrow([(250,204),(313,204),(313,246),(350,246)],P['blue_deep'],lw=2.1)
txt(303,228,r'$I$',ha='right',color=P['blue_deep'],bg='white')
arrow([(145,256),(145,306)],P['ink'],lw=1.9,dash=(0,(4,3)))
txt(158,282,'Extract once',ha='left',color=P['muted'])

box(25,306,240,370,'white',P['edge'],lw=1.1)
txt(145,328,r'Fixed evidence $E$',bold=True,color=P['muted'])
for name,label,x,y in [
    ('masks.png','Masks',35,356),('depth.png','Depth',155,356),
    ('albedo.png','Albedo',35,466),('shading.png','Shading',155,466),
    ('environment.png','HDR',35,576)]:
    photo(name,x,y,100,65)
    txt(x+50,y+82,label,color=P['muted'])
ax.add_patch(Polygon([(163,628),(186,592),(247,603),(223,637)],
                     fc=P['green_card'],ec=P['green'],lw=1.3*SCALE,zorder=3))
arrow([(207,617),(215,579)],P['green'],lw=1.7)
txt(242,588,r'$\mathbf{n}$',color=P['green'])
txt(205,658,'Planes',color=P['muted'])
txt(145,694,'Fixed across rounds',italic=True,color=P['muted'])
arrow([(265,453),(311,453),(311,388),(350,388)],P['blue_deep'],lw=2.1)
txt(301,421,r'$E$',ha='right',color=P['blue_deep'],bg='white')

# Three visual anchors. Large typography is reserved for these titles.
core(350,160,350,300,'Generator')
core(825,160,450,350,r'Memory $M_t$',memory=True)
core(1400,160,350,300,'Verifier')

# Reasoning remains a process within the generator; it is not memory.
txt(525,249,'Scene modeling',bold=True)
txt(525,282,'Geometry · materials')
line([(370,310),(680,310)],P['edge'],lw=1.3)
box(370,330,310,105,'#DDE8EE',z=2)
txt(525,353,'Light reasoning',bold=True,color=P['blue_deep'])
txt(525,389,'Source identity · type')
txt(525,422,'Pose · energy · color')

# Shared state and diagnostic history occupy the central core module.
txt(1050,249,r'Persistent light state $S_t$',bold=True)
txt(1050,282,'Stable IDs · types · parameters')
txt(1050,315,'Validated source updates')
line([(845,339),(1255,339)],P['edge'],lw=1.3)
txt(1050,378,r'Diagnostic history $H_t$',bold=True)
txt(1050,411,'Render / score archive')
txt(1050,444,'Object-wise advice')
txt(1050,477,'Implementation + outcomes')

# Numerical rubric and VLM are secondary components within the verifier.
box(1420,231,310,88,P['cream'],z=2)
txt(1575,248,'Rubric scoring',bold=True,color=P['ochre'])
txt(1575,277,'IoU · depth · surface planes')
txt(1575,306,'Albedo · shading · RGB')
arrow([(1575,319),(1575,351)],P['blue_deep'],lw=2)
txt(1575,375,'VLM diagnosis',bold=True,color=P['blue_deep'])
txt(1575,409,'Edits + history consistency')

# Short, prominent memory links: reads outward, validated writes inward.
arrow([(825,255),(700,255)],P['green'],lw=3.3,dash=(0,(4,3)),z=6)
txt(762.5,234,r'Read $M_t$',color=P['green'],bg='white')
arrow([(700,385),(760,385),(760,315),(825,315)],P['green'],lw=3.3,z=6)
txt(791,294,r'$\Delta S_t$',color=P['green'],bg='white')
arrow([(1275,255),(1400,255)],P['green'],lw=3.3,dash=(0,(4,3)),z=6)
txt(1337.5,234,r'Read $H_t$',color=P['green'],bg='white')
arrow([(1400,390),(1337,390),(1337,420),(1275,420)],P['green'],lw=3.3,z=6)
txt(1337,352,'Scores +\nreview',color=P['green'],bg='white')

# Feedback is the third edge of the core interaction, outside the modules.
txt(1050,35,r'Shared memory across $N$ iterations',italic=True,color=P['muted'])
arrow([(1575,160),(1575,88),(525,88),(525,160)],P['red'],
      lw=3.2,dash=(0,(5,3)),z=6)
txt(1050,88,'Next-round edit advice',color=P['red'],bg='white')

# Code execution and rendering are a supporting bridge below the three cores.
box(825,560,450,215,P['support'],P['edge'],lw=1.1)
txt(1050,582,'Blender / Cycles execution',bold=True,color=P['muted'])
box(844,611,125,125,'white',P['edge'],lw=1.1)
txt(906.5,635,r'Code $C_t$',color=P['muted'])
for y,length,offset in [(661,73,0),(675,53,12),(689,63,12)]:
    line([(861+offset,y),(861+offset+length,y)],P['muted'],lw=2)
txt(906.5,721,'Validated',italic=True,color=P['muted'])
arrow([(969,674),(1051.2,674)],P['blue_deep'],lw=2)
photo('render.png',1015,610,235,122)
txt(1132.5,752,r'Current render $\hat I_t$',color=P['muted'])

arrow([(525,460),(525,610),(825,610)],P['blue_deep'],lw=2.2,z=5)
txt(681,610,r'Code $C_t$',color=P['blue_deep'],bg='white')
arrow([(1275,610),(1575,610),(1575,460)],P['blue_deep'],lw=2.2,z=5)
txt(1430,610,'RGB + passes',color=P['blue_deep'],bg='white')
arrow([(1050,560),(1050,510)],P['green'],lw=2.5,z=6)
txt(1066,533,'Archive render',ha='left',color=P['green'],bg='white')

# Results stay small and unboxed, serving as examples rather than headline panels.
for label,x in [('initial',420),('best',1470)]:
    r=examples[label]
    title=f'Initial · round {r["round"]}' if label=='initial' else f'Best · round {r["round"]}'
    txt(x+105,643,title,color=P['muted'])
    photo(r['file'],x,663,210,109)
    txt(x+105,794,rf'$\mathcal{{L}}={r["loss"]:.3f}$',color=P['muted'])

for x,color,dash,label in [
    (42,P['blue_deep'],None,'Forward / render'),
    (470,P['red'],(0,(5,3)),'Edit feedback'),
    (855,P['ink'],(0,(4,3)),'Initialize'),
    (1135,P['green'],(0,(4,3)),'Read memory'),
    (1490,P['green'],None,'Write memory')]:
    arrow([(x,825),(x+44,825)],color,lw=2.4,dash=dash)
    txt(x+56,825,label,ha='left',color=P['muted'])

metadata={'Title':'Generator, shared memory, and verifier for scene reconstruction',
          'Author':'', 'Subject':'Three-core method overview derived from exp_1113_1445_rule'}
for ext in ('svg','pdf','png'):
    opts={'dpi':600} if ext=='png' else {}
    fig.savefig(HERE/f'lightproxy_method_v3.{ext}',facecolor='white',
                metadata=metadata if ext=='pdf' else None,**opts)
fig.savefig(HERE/'lightproxy_method_v3_preview.png',dpi=230,facecolor='white')

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
bounds=[]
outside=[]
for t in texts:
    b=t.get_window_extent(renderer).transformed(ax.transData.inverted())
    x0,y0,x1,y1=min(b.x0,b.x1),min(b.y0,b.y1),max(b.x0,b.x1),max(b.y0,b.y1)
    bounds.append((t.get_text(),x0,y0,x1,y1))
    if x0<0 or x1>W or y0<0 or y1>H:outside.append(t.get_text())
overlaps=[]
for i,(name,x0,y0,x1,y1) in enumerate(bounds):
    for other,a0,b0,a1,b1 in bounds[i+1:]:
        if min(x1,a1)-max(x0,a0)>1 and min(y1,b1)-max(y0,b0)>1:
            overlaps.append([name,other])
report={'size_mm':[180,84],'font_sizes_pt':[F_MAIN,F_SUB],
        'large_type_reserved_for':['Generator','Memory','Verifier'],
        'out_of_canvas_labels':outside,'overlapping_text_boxes':overlaps,
        'memory_read_targets':['Generator','Verifier'],
        'memory_write_sources':['Validated light-state update','Render archive','Rubric scores and VLM diagnosis'],
        'example_rounds':{k:v['round'] for k,v in examples.items()},
        'note':'Memory is the central visual anchor; execution and examples are supporting elements.'}
(HERE/'layout_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))


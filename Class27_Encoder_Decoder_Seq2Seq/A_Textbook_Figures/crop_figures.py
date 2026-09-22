import subprocess, re, sys, html
from xml.etree import ElementTree as ET
OVERRIDE={'J14_22':172,'J14_9':196,'J14_20':540,'J14_5':458,'J14_16':500,'J14_18':512,'G10_9':150,'G10_12':200,'J14_21':280,'J13_6':104,'J13_7':200,'J14_19':104}
def crop(pdf, page, figlabel, out, extra_top=6, extra_bot=4):
    xml=subprocess.run(['pdftotext','-bbox-layout','-f',str(page),'-l',str(page),pdf,'-'],capture_output=True,text=True).stdout
    root=ET.fromstring(xml.replace('xmlns="http://www.w3.org/1999/xhtml"',''))
    pg=root.find('.//page'); W=float(pg.get('width')); H=float(pg.get('height'))
    lines=[]
    for ln in root.iter('line'):
        words=[w.text for w in ln.iter('word')]
        lines.append((float(ln.get('yMin')),float(ln.get('yMax')),float(ln.get('xMin')),float(ln.get('xMax')),' '.join(words)))
    lines.sort()
    cap=[l for l in lines if (l[4]==figlabel.strip() or l[4].startswith(figlabel.strip()+' ')) and l[2]<0.35*W]
    assert cap, (page,figlabel)
    cy=cap[0][0]
    # figure spans from the caption up to the nearest 'body' line above it
    import re, statistics
    long=[l for l in lines if len(l[4].split())>=6 and (l[3]-l[2])>0.45*W]
    margin=statistics.median([l[2] for l in long]) if long else 0.2*W
    body=[l for l in lines if l[1] < cy-2 and ((len(l[4].split())>=6 and (l[3]-l[2])>0.45*W)
          or (abs(l[2]-margin)<3 and len(l[4].split())>=3)
          or re.search(r'\(\s*\d+\.\d+\s*\)\s*$',l[4]) or l[0]<100)]
    top = max([l[1] for l in body]+[40.0])
    # left/right extent of the figure: words between top and caption
    fw=[l for l in lines if top<=l[0] and l[1]<=cy-2]
    x0=min([l[2] for l in fw]+[W*0.15])-14; x1=max([l[3] for l in fw]+[W*0.85])+14
    x0=max(0,x0); x1=min(W,x1)
    # also stop at section headers
    y0=(top-extra_top) if top>100 else top+3; y1=cy-extra_bot
    name=out.split('/')[-1][:-4]
    if name in OVERRIDE: y0=OVERRIDE[name]; fw=[l for l in lines if y0<=l[0] and l[1]<=cy-2]; x0=max(0,min([l[2] for l in fw]+[W*0.15])-14); x1=min(W,max([l[3] for l in fw]+[W*0.85])+14)
    # widen generously, then tighten to the ink with an even margin, so a
    # framed figure keeps its frame and nothing drawn is cut
    x0=max(0,x0-30); x1=min(W,x1+30)
    above=[l[1] for l in lines if l[1] <= y0+3]                    # text lines ending above the region
    y0=max(max(above)+1.5 if above else 0, y0-14)
    if above and max(above) < 50: y0=max(y0, 52)                   # a running head and its rule
    floor=y0
    x0,y0,x1,y1=ink_bbox(pdf,page,x0,y0,x1,y1,pad=PAD); y0=max(y0,floor); y1=min(y1,cy-1)
    subprocess.run(['pdftocairo','-pdf','-f',str(page),'-l',str(page),'-x',str(int(x0)),'-y',str(int(y0)),'-W',str(int(x1-x0)),'-H',str(int(y1-y0)),'-paperw',str(int(x1-x0)),'-paperh',str(int(y1-y0)),pdf,out],check=True)
    print(out, 'y', round(y0), round(y1), 'x', round(x0), round(x1))


PAD=6
def ink_bbox(pdf,page,x0,y0,x1,y1,pad=6,dpi=144):
    """Rasterise the region and return the bounding box of its ink, in points."""
    import tempfile, os
    from PIL import Image
    import numpy as np
    tmp=tempfile.mkdtemp(); base=os.path.join(tmp,'r')
    subprocess.run(['pdftoppm','-r',str(dpi),'-f',str(page),'-l',str(page),'-x',str(int(x0*dpi/72)),'-y',str(int(y0*dpi/72)),
                    '-W',str(int((x1-x0)*dpi/72)),'-H',str(int((y1-y0)*dpi/72)),'-png',pdf,base],check=True)
    png=[f for f in os.listdir(tmp) if f.endswith('.png')][0]
    im=np.asarray(Image.open(os.path.join(tmp,png)).convert('L'))
    ink=np.argwhere(im<235)
    if len(ink)==0: return x0,y0,x1,y1
    (r0,c0),(r1,c1)=ink.min(0),ink.max(0)
    s=72.0/dpi
    return (max(0,x0+c0*s-pad), max(0,y0+r0*s-pad), x0+c1*s+pad, y0+r1*s+pad)
S='/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/SLP3_JurafskyMartin/book/ed3book_aug26.pdf'
G='/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/DeepLearningBook_GoodfellowBengioCourville/book/Deep_Learning_Goodfellow_Bengio_Courville.pdf'
jobs=[(S,314,'Figure 14.5','J14_5'),(S,320,'Figure 14.9','J14_9'),(S,327,'Figure 14.16','J14_16'),(S,329,'Figure 14.17','J14_17'),(S,329,'Figure 14.18','J14_18'),(S,331,'Figure 14.19','J14_19'),(S,331,'Figure 14.20','J14_20'),(S,332,'Figure 14.21','J14_21'),(S,333,'Figure 14.22','J14_22'),(S,294,'Figure 13.6','J13_6'),(S,295,'Figure 13.7','J13_7'),
      (G,396,'Figure 10.4:','G10_4'),(G,399,'Figure 10.6:','G10_6'),(G,408,'Figure 10.9:','G10_9'),(G,409,'Figure 10.10:','G10_10'),(G,412,'Figure 10.12:','G10_12')]
for pdf,pg,lab,name in jobs:
    try: crop(pdf,pg,lab,'/tmp/crops27/%s.pdf'%name)
    except Exception as e: print('FAIL',name,e)

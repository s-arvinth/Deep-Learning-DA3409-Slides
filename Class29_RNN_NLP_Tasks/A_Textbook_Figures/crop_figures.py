import subprocess, re, sys, html
from xml.etree import ElementTree as ET
OVERRIDE={'G10_11':115,'G10_14':170,'G10_3':225,'J14_15':370,'J14_12':100,'J14_8':456,'J14_10':105,'J14_9':196,'G10_7':118,'G10_13':100,'J14_11':186}
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
    subprocess.run(['pdftocairo','-pdf','-f',str(page),'-l',str(page),'-x',str(int(x0)),'-y',str(int(y0)),'-W',str(int(x1-x0)),'-H',str(int(y1-y0)),'-paperw',str(int(x1-x0)),'-paperh',str(int(y1-y0)),pdf,out],check=True)
    print(out, 'y', round(y0), round(y1))
S='/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/SLP3_JurafskyMartin/book/ed3book_aug26.pdf'
G='/sessions/compassionate-admiring-einstein/mnt/TA-Duty/Reference-Textbooks/DeepLearningBook_GoodfellowBengioCourville/book/Deep_Learning_Goodfellow_Bengio_Courville.pdf'
jobs=[(S,316,'Figure 14.6','J14_6'),(S,318,'Figure 14.7','J14_7'),(S,318,'Figure 14.8','J14_8'),(S,320,'Figure 14.9','J14_9'),
      (S,321,'Figure 14.10','J14_10'),(S,322,'Figure 14.11','J14_11'),(S,323,'Figure 14.12','J14_12'),(S,326,'Figure 14.15','J14_15'),
      (G,394,'Figure 10.3:','G10_3'),(G,404,'Figure 10.7:','G10_7'),(G,404,'Figure 10.8:','G10_8'),(G,410,'Figure 10.11:','G10_11'),
      (G,415,'Figure 10.13:','G10_13'),(G,416,'Figure 10.14:','G10_14')]
for pdf,pg,lab,name in jobs:
    try: crop(pdf,pg,lab,'/tmp/crops29/%s.pdf'%name)
    except Exception as e: print('FAIL',name,e)

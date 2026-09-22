import subprocess, re, sys, html
from xml.etree import ElementTree as ET
OVERRIDE={'G10_17':150,'J14_14':308,'J14_22':172,'J14_9':196,'J14_20':540,'J14_5':458,'J14_16':500,'J14_18':512,'G10_9':150,'G10_12':200,'J14_21':280,'J13_6':104,'J13_7':200,'J14_19':104}
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
jobs=[(G,418,'Figure 10.15:','G10_15'),(G,425,'Figure 10.16:','G10_16'),(G,430,'Figure 10.17:','G10_17'),(G,433,'Figure 10.18:','G10_18'),
      (S,325,'Figure 14.13','J14_13'),(S,325,'Figure 14.14','J14_14')]
for pdf,pg,lab,name in jobs:
    try: crop(pdf,pg,lab,'/tmp/crops28/%s.pdf'%name)
    except Exception as e: print('FAIL',name,e)

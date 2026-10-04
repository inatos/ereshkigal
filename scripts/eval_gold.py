#!/usr/bin/env python3
"""Gold metrics: accuracy, global BA, mean-family BA."""
import json, sys
from pathlib import Path
from collections import defaultdict

def ba(pred, gold, n):
    tp = [0.0]*n; sup=[0.0]*n
    for p,g in zip(pred,gold):
        if 0<=g<n:
            sup[g]+=1
            if p==g: tp[g]+=1
    rec=[tp[c]/sup[c] for c in range(n) if sup[c]>0]
    return sum(rec)/len(rec) if rec else 0.0

def main(gold_path, pred_path):
    gold=[json.loads(l) for l in Path(gold_path).read_text().splitlines() if l.strip()]
    pred=[json.loads(l) for l in Path(pred_path).read_text().splitlines() if l.strip()]
    gby={r['id']:r for r in gold}
    n=max(len(r['options']) for r in gold)
    pred_i, gold_i, fams=[],[],[]
    for p in pred:
        g=gby[p['id']]
        pr=p['probabilities']
        pred_i.append(max(range(len(pr)), key=lambda i: pr[i]))
        gold_i.append(int(g['label']))
        fams.append(g.get('family','all'))
    acc=sum(a==b for a,b in zip(pred_i,gold_i))/len(pred_i)
    global_ba=ba(pred_i,gold_i,n)
    by=defaultdict(lambda: ([],[]))
    groups=defaultdict(list)
    for f,p,g,row in zip(fams,pred_i,gold_i,pred):
        by[f][0].append(p); by[f][1].append(g)
        gid=gby[row['id']].get('group_id')
        if gid:
            groups[gid].append(p==g)
    fam_ba=sum(ba(ps,gs,n) for ps,gs in by.values())/len(by)
    allc=sum(1 for hits in groups.values() if hits and all(hits))/len(groups) if groups else None
    out={"n":len(pred_i),"accuracy":acc,"global_ba":global_ba,"family_ba":fam_ba,
         "group_all_correct":allc,
         "families":{k:ba(v[0],v[1],n) for k,v in by.items()}}
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])

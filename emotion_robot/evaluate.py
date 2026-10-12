import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .model import Model
from .train import load_split, scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',default='artifacts/test.json')
    args=p.parse_args()
    model=Model(args.model)
    root=Path(args.data)
    tx,ax,y=load_split(root/'test_sent_emo.csv',root/'test')
    tp=softmax(tx@model.tw.T+model.tb,axis=1)
    ap=softmax(((ax-model.mean)/model.scale)@model.aw.T+model.ab,axis=1)
    alpha=model.meta['audio_weight'] * (ax[:,40]>=.003)[:,None]
    rng=np.random.default_rng(42)
    report={'rows':len(y), 'model_metadata':model.meta, 'learned_parameters':model.parameter_count,
            'text':scores(y,tp),'audio':scores(y,ap),'fusion':scores(y,(1-alpha)*tp+alpha*ap),
            'shuffled_audio':scores(y,(1-alpha)*tp+alpha*ap[rng.permutation(len(ap))])}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['macro_f1'] for k,v in report.items() if isinstance(v,dict) and 'macro_f1' in v},indent=2))


if __name__=='__main__': main()

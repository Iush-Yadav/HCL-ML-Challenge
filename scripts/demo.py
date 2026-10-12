"""Synthetic integration fixture, NOT evidence of emotion recognition accuracy."""
import base64
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from emotion_robot.features import LABELS, text_features, audio_features
from emotion_robot.train import fit
from emotion_robot.model import Model
from emotion_robot.stream import Session

TEXTS=['This is frustrating', 'That is unpleasant', 'I am worried', 'What wonderful news',
       'The meeting is at noon', 'I miss my friend', 'I did not expect that']


def tone(label, seed):
    rng=np.random.default_rng(seed)
    t=np.arange(16000)/16000
    # Artificial class-correlated tones exercise data plumbing only.
    x=(.05+.025*label)*np.sin(2*np.pi*(130+label*100+rng.normal(0,4))*t)
    x+=rng.normal(0,.005,len(t))
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((x*32767).astype('<i2').tobytes())
    return out.getvalue()


def fixture(seed):
    tx,ax,y=[],[],[]
    for label in range(7):
        for k in range(8):
            tx.append(text_features(TEXTS[label]))
            ax.append(audio_features(tone(label,seed+k))[0]); y.append(label)
    return np.array(tx),np.array(ax),np.array(y)


def event(seq=0,label=3):
    return {'seq':seq,'start_ms':seq*1200,'end_ms':seq*1200+1000,
            'text':TEXTS[label],'audio_b64':base64.b64encode(tone(label,99)).decode()}


def main():
    path=Path('models/synthetic.npz')
    fit(fixture(0),fixture(100),path,{'kind':'SYNTHETIC_INTEGRATION_ONLY','warning':'Not trained on MELD'})
    # Force a nonzero audio contribution in the synthetic integration fixture.
    # Production training uses the dev-selected value, without this override.
    with np.load(path,allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    meta=json.loads(str(arrays['metadata'])); meta['audio_weight']=.5
    meta['fixture_override']='Equal fusion to exercise both branches; not dev-selected'
    arrays['metadata']=json.dumps(meta); np.savez_compressed(path,**arrays)
    model=Model(path); session=Session(model)
    Path('artifacts/demo').mkdir(parents=True, exist_ok=True)
    Path('artifacts/demo/input.jsonl').write_text(json.dumps(event())+'\n')
    results=session.process(event())
    Path('artifacts/demo/output.json').write_text(json.dumps(results,indent=2))
    timings=[]
    for i in range(1,101):
        start=time.perf_counter(); session.process(event(i)); timings.append((time.perf_counter()-start)*1000)
    chip=subprocess.run(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True).stdout.strip() if sys.platform=='darwin' else platform.processor()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report={'scope':'Synthetic one-second WAV, 100 warm calls; includes fixture generation; no MELD accuracy claim',
            'hardware':chip or 'CPU model unavailable in sandbox; see docs/validation.md','platform':platform.platform(),'python':platform.python_version(),
            'p50_ms':float(np.percentile(timings,50)),'p95_ms':float(np.percentile(timings,95)),
            'max_ms':max(timings),'peak_process_rss_mb':peak/(1024**2 if sys.platform=='darwin' else 1024),
            'rss_scope':'Includes imports and synthetic training',
            'learned_parameters':model.parameter_count,'model_bytes':path.stat().st_size}
    Path('artifacts/demo/benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(results,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()

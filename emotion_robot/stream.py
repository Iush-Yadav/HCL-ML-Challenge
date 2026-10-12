"""Bounded, ordered utterance stream. JSONL in; state and response events out."""
import argparse
import base64
import binascii
import json
import sys
import time

import numpy as np

from .features import LABELS
from .model import Model, respond


class Session:
    def __init__(self, model):
        self.model = model
        self.last_seq = -1
        self.last_end = 0.

    def process(self, event):
        start = time.perf_counter()
        seq = event.get('seq')
        if type(seq) is not int or seq <= self.last_seq:
            raise ValueError('seq must be an increasing integer')
        begin, end = event.get('start_ms'), event.get('end_ms')
        if any(type(v) not in (int,float) or not np.isfinite(v) for v in (begin,end)):
            raise ValueError('start_ms and end_ms must be finite numbers')
        if begin < self.last_end or not 200 <= end-begin <= 15000:
            raise ValueError('utterance time ranges must be ordered, non-overlapping, and 200–15000 ms')
        encoded = event.get('audio_b64')
        if not isinstance(encoded,str) or len(encoded) > 10_700_000:
            raise ValueError('audio_b64 must be base64 WAV, at most 8 MB decoded')
        try: raw = base64.b64decode(encoded, validate=True)
        except (ValueError,binascii.Error) as exc: raise ValueError('invalid base64 audio') from exc
        text = event.get('text')
        p, tp, ap, quality, alpha = self.model.predict(text, raw)
        if abs(quality['duration_s']*1000-(end-begin)) > 150:
            raise ValueError('timestamps disagree with WAV duration by more than 150 ms')
        emotion = LABELS[int(p.argmax())]
        disagreement = int(tp.argmax()) != int(ap.argmax()) and not quality['silent']
        uncertain = bool(p.max() < .5 or disagreement or quality['silent'])
        # Commit only after all validation and inference succeed.
        self.last_seq, self.last_end = seq, end
        state = {'type': 'state', 'seq': seq, 'emotion': emotion,
                 'scores': dict(zip(LABELS, map(float,p))),
                 'confidence': float(p.max()), 'uncertain': uncertain,
                 'tags': [tag for tag,yes in [('modalities_disagree',disagreement),
                            ('audio_unusable',quality['silent']),('tentative',uncertain)] if yes],
                 'evidence': {'text_emotion': LABELS[int(tp.argmax())], 'audio_emotion': LABELS[int(ap.argmax())],
                              'audio_weight': alpha, 'audio_quality': quality},
                 'model_provenance': self.model.meta['provenance'],
                 'latency_ms': round((time.perf_counter()-start)*1000,3)}
        return state, {'type': 'response', 'seq': seq, 'text': respond(text,emotion,uncertain)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    session = Session(Model(args.model))
    # Read a bounded line to prevent unbounded buffering of client input.
    while True:
        line = sys.stdin.buffer.readline(10_800_001)
        if not line: break
        if len(line)>10_800_000:
            print(json.dumps({'type':'error','message':'event too large; closing stream'}),flush=True)
            break
        try:
            event = json.loads(line)
            if not isinstance(event,dict): raise ValueError('event must be an object')
            for output in session.process(event): print(json.dumps(output),flush=True)
        except (ValueError,KeyError,TypeError) as exc:
            print(json.dumps({'type':'error','message':str(exc)}),flush=True)


if __name__ == '__main__': main()

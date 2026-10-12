"""Produce one event for piping to the persistent JSONL inference process."""
import argparse
import base64
import json
import wave
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--wav',required=True); p.add_argument('--text',required=True)
p.add_argument('--seq',type=int,default=0); p.add_argument('--start-ms',type=float,default=0)
a=p.parse_args()
with wave.open(a.wav,'rb') as w: duration=w.getnframes()/w.getframerate()*1000
print(json.dumps({'seq':a.seq,'start_ms':a.start_ms,'end_ms':a.start_ms+duration,
                  'text':a.text,'audio_b64':base64.b64encode(Path(a.wav).read_bytes()).decode()}))

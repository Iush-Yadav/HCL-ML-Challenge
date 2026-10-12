"""Convert already extracted official MELD clips; never silently skip a row."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--clips',required=True,help='split-specific directory containing diaN_uttN.mp4')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not shutil.which('ffmpeg'): raise SystemExit('Install ffmpeg first')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    with open(args.csv,newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key=f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
            src=Path(args.clips)/f'{key}.mp4'
            if not src.exists(): raise FileNotFoundError(src)
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(src),'-vn','-ac','1',
                            '-ar','16000','-c:a','pcm_s16le',str(out/f'{key}.wav')],check=True)


if __name__=='__main__': main()

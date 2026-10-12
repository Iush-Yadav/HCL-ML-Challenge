"""Train on official MELD splits; select fusion weight using dev only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from .features import LABELS, text_features, audio_features


def load_split(csv_path, audio_dir):
    texts, acoustics, labels, ids = [], [], [], []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
            raw = (Path(audio_dir) / f'{key}.wav').read_bytes()
            feat, _ = audio_features(raw)
            texts.append(text_features(row['Utterance']))
            acoustics.append(feat)
            labels.append(LABELS.index(row['Emotion'].lower()))
            ids.append(key)
    if not texts or len(ids) != len(set(ids)):
        raise ValueError('split is empty or has duplicate utterance IDs')
    return np.array(texts), np.array(acoustics), np.array(labels)


def scores(y, p):
    pred = p.argmax(1)
    return {'macro_f1': float(f1_score(y, pred, labels=range(7), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=range(7), average='weighted', zero_division=0)),
            'accuracy': float(np.mean(pred == y)),
            'per_class': classification_report(y, pred, labels=range(7), target_names=LABELS,
                                                output_dict=True, zero_division=0)}


def fit(train, dev, output, provenance):
    tx, ax, y = train
    dt, da, dy = dev
    if set(y) != set(range(7)):
        raise ValueError('training split must contain all seven MELD emotions')
    scaler = StandardScaler().fit(ax)
    tm = LogisticRegression(C=2, class_weight='balanced', max_iter=500, random_state=42).fit(tx, y)
    am = LogisticRegression(C=.1, class_weight='balanced', max_iter=500, random_state=42).fit(scaler.transform(ax), y)
    tp, ap = tm.predict_proba(dt), am.predict_proba(scaler.transform(da))
    usable = (da[:,40] >= .003)[:,None]
    candidates = [(float(a), scores(dy, (1-a*usable)*tp+a*usable*ap)['macro_f1']) for a in np.linspace(0, 1, 11)]
    alpha = max(candidates, key=lambda pair: pair[1])[0]
    metadata = {'labels': LABELS, 'audio_weight': alpha, 'provenance': provenance,
                'training_rows': len(y), 'dev_rows': len(dy), 'seed': 42,
                'feature_version': 1, 'generator': 'deterministic templates; zero learned parameters'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, tw=tm.coef_, tb=tm.intercept_, aw=am.coef_, ab=am.intercept_,
                        mean=scaler.mean_, scale=scaler.scale_, metadata=json.dumps(metadata))
    report = {'metadata': metadata, 'fusion_search_dev_macro_f1': candidates,
              'text': scores(dy,tp), 'audio': scores(dy,ap), 'fusion': scores(dy,(1-alpha*usable)*tp+alpha*usable*ap)}
    output.with_suffix('.dev.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='root containing CSV files and train/dev WAV folders')
    parser.add_argument('--output', default='models/meld.npz')
    args = parser.parse_args()
    root = Path(args.data)
    provenance = {'kind': 'MELD', 'source': 'https://github.com/declare-lab/MELD',
                  'csv_sha256': {s: hashlib.sha256((root/f'{s}_sent_emo.csv').read_bytes()).hexdigest()
                                 for s in ('train','dev')}}
    report = fit(load_split(root/'train_sent_emo.csv', root/'train'),
                 load_split(root/'dev_sent_emo.csv', root/'dev'), args.output, provenance)
    print(json.dumps({k: v['macro_f1'] for k,v in report.items() if k in ('text','audio','fusion')}, indent=2))


if __name__ == '__main__': main()

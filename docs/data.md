# MELD data and experiments

Run these commands from the repository root.

Sources: [official repository](https://github.com/declare-lab/MELD), [official download](https://huggingface.co/datasets/declare-lab/MELD), [paper](https://aclanthology.org/P19-1050/). Obtain the original train/dev/test CSVs from the repository's `data/MELD` folder and the matching clips from `MELD.Raw.tar.gz`. Review the dataset's terms before redistribution; Friends media is not included in this repository.

Extract the archive locally and identify the train, dev, and test clip directories. Their names differ from the normalized layout below. Install FFmpeg separately for preparation; it is not used at inference time.

```sh
mkdir -p data/meld
# Copy train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv into data/meld.
python scripts/prepare_meld.py --csv data/meld/train_sent_emo.csv --clips /path/to/train_clips --output data/meld/train
python scripts/prepare_meld.py --csv data/meld/dev_sent_emo.csv --clips /path/to/dev_clips --output data/meld/dev
python scripts/prepare_meld.py --csv data/meld/test_sent_emo.csv --clips /path/to/test_clips --output data/meld/test
python -m emotion_robot.train --data data/meld --output models/meld.npz
python -m emotion_robot.evaluate --data data/meld --model models/meld.npz
```

Expected layout: `data/meld/{train,dev,test}/diaN_uttN.wav` alongside the three CSVs. Split prefixes matter: dialogue IDs repeat across official splits. The loader fails on missing, malformed, duplicate, or out-of-contract records instead of silently changing the evaluation population. Currently audio is restricted to 0.2–15 seconds; audit MELD durations before the full run and explicitly document any filtering or expanded duration contract. Do not silently truncate clips. Preparation preserves their full duration.

Train text and acoustic logistic classifiers on **train only**; fit acoustic normalization there too. Select the fusion coefficient by dev macro-F1 over a fixed 0.0–1.0 grid, with lower audio weight winning ties. Store CSV hashes, provenance, and the selected coefficient in the model. Run test once after choices are frozen. `artifacts/test.json` includes macro-F1, weighted F1, accuracy, per-class statistics, text/audio ablations, and shuffled audio. Compare fusion against text-only and shuffled audio before claiming multimodal value. If dev chooses 0 or 1, the resulting model is unimodal in practice and does **not** establish the challenge's multimodal objective; investigate features within the timebox rather than conceal that result.


# Emotion Robot

A local text + audio baseline for the MELD challenge. It takes an utterance transcript and an aligned WAV recording, predicts an emotion, and returns a short response grounded in the input.

**Current status:** the synthetic demo and integration tests work. MELD training, held-out evaluation, and a real-speech demonstration are still pending.

## Quick start

Run from the repository root with Python 3.11+:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make demo
make test
make replay
```

The demo uses artificial tones and sample text to exercise the pipeline. It does not measure emotion-recognition accuracy. Generated inputs and results go into `artifacts/demo/`; model files go into `models/`. Both directories are ignored by Git. The exact dependency versions used for the recorded run are in `requirements-lock.txt`.

Without Make, use `python scripts/demo.py` and `python -m unittest discover -s tests -v`.

## Repository layout

```text
emotion_robot/       Feature extraction, training, inference, and evaluation
scripts/             Data preparation, synthetic demo, and input helper
tests/              Pipeline and input-validation tests
docs/
   data.md           MELD setup and experiment commands
   design.md         Features, parameter budget, and streaming contract
   validation.md     Checks run and known limitations
   attribution.md    External components and AI assistance
   results/          Saved results from the documented synthetic run
Makefile             Common development commands
requirements.txt     Runtime dependencies
requirements-lock.txt  Versions used for local verification
```

## Input and output

Keep `python -m emotion_robot.stream --model models/meld.npz` running and send one JSON line per completed utterance. Each input contains a sequence number, start/end timestamps, a transcript, and base64-encoded PCM WAV audio. The process emits a `state` event followed by a `response` event. Wait for both before sending the next utterance.

Once a MELD model has been trained, send a recording with:

```sh
python scripts/send_utterance.py --wav your.wav --text "I did not expect that" \
  | python -m emotion_robot.stream --model models/meld.npz
```

The state includes a MELD emotion, uncalibrated scores, audio quality, and modality disagreement. The response quotes the transcript and uses tentative language when the signals conflict or confidence is low. See [the interface contract](docs/design.md#real-time-contract) for limits and error handling.

## Approach

Two small logistic classifiers use hashed text features and acoustic statistics. Their scores are fused using a weight selected on the MELD dev split. Responses use deterministic templates. The complete local path contains **29,101 fitted scalars**, including normalization statistics and the fusion coefficient.

The latency target is p95 below 250 ms after a completed utterance arrives. The recorded one-second synthetic benchmark was 0.85 ms p95 on an Apple M5 with 16 GiB RAM; peak process RSS was about 139 MiB. These measurements exclude capture, transcription, and speech playback. Full details are in the [validation record](docs/validation.md).

## Remaining work

Follow [the MELD setup guide](docs/data.md), then run `make train` and `make evaluate`. Audit clip durations and missing media first. Compare fused, text-only, audio-only, and shuffled-audio scores before claiming that the second modality helps.

A completed submission still needs a real held-out utterance demonstrated through both outputs, test-set results, real-media latency, and error analysis. Vision, reinforcement learning, ASR, TTS, microphone capture, and a web UI are outside this baseline's scope.

The current features can confuse speaker identity, music, laughter, and overlapping speech with emotion. Responses are repetitive and only shallowly grounded. MELD sitcom results would not establish performance with robot users.

See [attribution](docs/attribution.md) for dependencies and AI-assisted development disclosure.

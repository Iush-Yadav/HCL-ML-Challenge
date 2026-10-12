# Validation record — 2026-10-13

This record covers the synthetic integration fixture only. **No MELD training, MELD test score, or real-speech accuracy has been measured.**

Hardware: Apple M5, 16 GiB installed RAM, macOS 26.6.2 arm64. CPU-only; no GPU used. Python 3.14.7. Exact library versions are in `../requirements-lock.txt`. CPU and RAM were read using `sysctl`; no hardware values were inferred from timings.

Verified locally:

- `python scripts/demo.py`: creates numeric model, aligned WAV/transcript event, emotion state, response, and timing report.
- `python -m unittest discover -s tests -v`: eight passing tests covering output structure/parameter count, text influence, audio influence, duplicate rejection and recovery, timestamp validation without state corruption, invalid inputs, silence fallback, and malformed audio.
- `python -m emotion_robot.stream --model models/synthetic.npz < artifacts/demo/input.jsonl`: emits a `joy` state and a response quoting “What wonderful news”. This is an artificial tone/text pair, not a recognized human emotion.

The recorded benchmark runs 100 warm calls on a one-second synthetic WAV. [recorded benchmark](results/benchmark.json) reports latency and peak process resident memory, including imported dependencies and synthetic training. In the initial run, p95 was 0.85 ms and peak RSS was 138.8 MiB. This does not benchmark live transcription, endpointing, audio capture, cold startup, TTS, full 15-second inputs, or a loaded robot controller. Actual timing varies by run.

The model stores 29,101 fitted scalars including normalization and the fusion coefficient. Compressed synthetic model size is roughly 6.4 KB; a real trained model can compress differently. Dense text matrices for full MELD training require considerably more memory than inference (about 312 MiB for 9,989 × 4,096 float64 values, before copies and optimizer overhead). Full training resources remain unmeasured.

Uncertainty heuristics are not calibrated: max fused score below 0.5, different modality argmax labels, or silent audio cause a tentative response. Silence threshold is mean RMS below 0.003 on normalized PCM. These thresholds require real-data validation. Clipping is reported but not yet used to gate inference. Test and dev fusion evaluation apply the same silence mask as inference; audio-only evaluation measures the acoustic classifier without that fallback.

Outstanding: audit MELD duration compatibility and complete media availability, run official splits, validate class performance and audio contribution, demonstrate a held-out real utterance, and benchmark real-media input. Current code intentionally fails on incompatible media rather than report misleading partial-dataset metrics. The official archive listing was inspected; the approximately 10.9 GB media archive was not downloaded.

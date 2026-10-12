# Design notes

```text
transcript → hashed unigram/bigram vector → text linear classifier ─┐
                                                                 ├→ weighted probabilities
WAV → frame energy / zero crossings / spectral features → audio ──┘
    → emotion + uncertainty + evidence → response template → JSONL
```

Text has 4,096 hashed features. Audio uses 25 ms windows at 10 ms hops: means/stds of 20 log spectral bands, RMS, zero-crossing rate, and spectral centroid (46 features). Both classifiers use class balancing. Hashing, resampling, FFTs, and templates are deterministic and contain no learned parameters.

| Component | Count |
| --- | ---: |
| Text weights + biases | 28,679 |
| Audio weights + biases | 329 |
| Fitted normalization mean + scale (conservatively counted) | 92 |
| Selected fusion coefficient | 1 |
| **Total** | **29,101** |

No hidden backbone, projector, auxiliary generator, or pretrained encoder is required. The count is total stored fitted scalars, not trainable or active parameters. This is well below 6 billion. Numeric NPZ loading disables pickle.

## Real-time contract

“Real-time” means a target of **p95 < 250 ms after a completed utterance reaches the warm local process**, for 0.2–15 second audio. This is a target, not a measured guarantee for all durations. The demo benchmarks one-second synthetic utterances only. End-to-end latency also includes utterance duration, upstream endpointing/transcription, transport, and robot speech rendering, none of which is benchmarked here. Caller-provided transcripts avoid hiding an ASR model outside the parameter budget.

Input fields: increasing integer `seq`, ordered non-overlapping `start_ms`/`end_ms`, nonempty `text` (at most 2,000 characters), and `audio_b64` containing mono/stereo 16-bit PCM WAV at 8–48 kHz. WAV duration must agree with timestamps within 150 ms. Maximum WAV size is 8 MB. The process handles one utterance at a time: the caller must wait for both outputs before sending the next to avoid queue growth. Flushes expose each completed event immediately; response text is emitted as one event, not token streaming.

`state` contains `emotion`, all seven `scores`, `confidence`, `uncertain`, `tags`, modality predictions, actual fusion weight, audio quality, provenance, and inference latency. Scores are **uncalibrated**, not reliable probabilities of a person's internal emotional state. Silence falls back to text with an explicit `audio_unusable` tag and tentative response. No smoothing is applied across turns: this prevents one mistaken label from persisting, but misses conversational context. Timestamps track ordering, not emotion history.


"""Non-learned feature extraction; accepts mono/stereo 16-bit PCM WAV."""
import io
import wave

import numpy as np
from scipy.signal import resample_poly
from sklearn.feature_extraction.text import HashingVectorizer

TEXT_DIM = 4096
AUDIO_DIM = 46
LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
HASHER = HashingVectorizer(n_features=TEXT_DIM, alternate_sign=False,
                           ngram_range=(1, 2), norm='l2', lowercase=True)


def text_features(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('text must contain 1–2000 characters')
    return HASHER.transform([text]).toarray()[0]


def audio_features(raw):
    if len(raw) > 8_000_000:
        raise ValueError('WAV exceeds 8 MB')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as wav:
            rate, channels = wav.getframerate(), wav.getnchannels()
            count = wav.getnframes()
            if wav.getsampwidth() != 2 or channels not in (1, 2):
                raise ValueError('use mono/stereo 16-bit PCM WAV')
            if not 8000 <= rate <= 48000 or not 0.2 <= count / rate <= 15:
                raise ValueError('audio must be 0.2–15 seconds at 8–48 kHz')
            samples = wav.readframes(count)
            if len(samples) != count * channels * 2:
                raise ValueError('truncated WAV')
    except (wave.Error, EOFError) as exc:
        raise ValueError('invalid PCM WAV') from exc
    x = np.frombuffer(samples, dtype='<i2').astype(float).reshape(-1, channels).mean(1) / 32768
    duration = len(x) / rate
    x = resample_poly(x, 16000, rate)
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    power = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    frequencies = np.fft.rfftfreq(512, 1 / 16000)
    centroid = (power @ frequencies) / (power.sum(1) + 1e-12) / 8000
    # Twenty log spectral bands retain coarse timbre; no learned encoder.
    edges = np.geomspace(80, 8000, 21)
    bands = np.array([np.log1p(power[:, (frequencies >= a) & (frequencies < b)].mean(1))
                      for a, b in zip(edges[:-1], edges[1:])]).T
    features = np.r_[bands.mean(0), bands.std(0), rms.mean(), rms.std(),
                     zcr.mean(), zcr.std(), centroid.mean(), centroid.std()]
    quality = {'duration_s': round(duration, 3), 'rms': round(float(rms.mean()), 5),
               'clipping_fraction': round(float(np.mean(np.abs(x) >= .99)), 5),
               'silent': bool(rms.mean() < .003)}
    return features, quality

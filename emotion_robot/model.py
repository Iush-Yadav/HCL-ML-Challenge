import json

import numpy as np
from scipy.special import softmax

from .features import AUDIO_DIM, TEXT_DIM, LABELS, text_features, audio_features


class Model:
    def __init__(self, path):
        # Numeric arrays only: never unpickle a downloaded model.
        with np.load(path, allow_pickle=False) as f:
            self.tw, self.tb = f['tw'], f['tb']
            self.aw, self.ab = f['aw'], f['ab']
            self.mean, self.scale = f['mean'], f['scale']
            self.meta = json.loads(str(f['metadata']))
        if self.tw.shape != (7, TEXT_DIM) or self.aw.shape != (7, AUDIO_DIM):
            raise ValueError('model feature dimensions do not match')
        if self.tb.shape != (7,) or self.ab.shape != (7,) or self.mean.shape != (AUDIO_DIM,) or self.scale.shape != (AUDIO_DIM,):
            raise ValueError('invalid model array shape')
        if any(not np.isfinite(a).all() for a in (self.tw,self.tb,self.aw,self.ab,self.mean,self.scale)) or np.any(self.scale <= 0):
            raise ValueError('invalid model values')
        if self.meta['labels'] != LABELS or not 0 <= self.meta['audio_weight'] <= 1:
            raise ValueError('invalid model metadata')

    @property
    def parameter_count(self):
        # Conservatively count fitted normalization statistics and fusion scalar.
        return sum(a.size for a in (self.tw,self.tb,self.aw,self.ab,self.mean,self.scale)) + 1

    def predict(self, text, raw, mode='fusion'):
        t = text_features(text)
        a, quality = audio_features(raw)
        tp = softmax(self.tw @ t + self.tb)
        ap = softmax(self.aw @ ((a-self.mean)/self.scale) + self.ab)
        alpha = self.meta['audio_weight']
        if mode == 'text' or quality['silent']: alpha = 0
        elif mode == 'audio': alpha = 1
        elif mode != 'fusion': raise ValueError('unknown inference mode')
        return (1-alpha)*tp + alpha*ap, tp, ap, quality, alpha


RESPONSES = {
    'anger': 'That sounds frustrating. What would help right now?',
    'disgust': 'That sounds unpleasant. Would you like to say more?',
    'fear': 'That sounds worrying. What feels most uncertain?',
    'joy': 'That sounds like good news. What are you most pleased about?',
    'neutral': 'I’m listening. What would you like to do next?',
    'sadness': 'That sounds difficult. Would you like to talk about it?',
    'surprise': 'That sounds unexpected. What happened next?',
}


def respond(text, emotion, uncertain):
    # Quoting ties output to the actual utterance without inventing facts.
    quote = ' '.join(text.split())[:100]
    prompt = 'I may be reading your tone wrong. How are you feeling?' if uncertain else RESPONSES[emotion]
    return f'You said, “{quote}”. {prompt}'

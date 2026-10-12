import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from emotion_robot.features import AUDIO_DIM, audio_features
from emotion_robot.model import Model
from emotion_robot.stream import Session
from emotion_robot.train import fit
from scripts.demo import fixture, event, tone


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.tmp.name)/'fixture.npz'
        fit(fixture(0),fixture(100),cls.path,{'kind':'SYNTHETIC_TEST'})
        cls.model=Model(cls.path)
        cls.model.meta['audio_weight']=.5

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_end_to_end(self):
        state,response=Session(self.model).process(event())
        self.assertEqual(state['type'],'state'); self.assertEqual(response['type'],'response')
        self.assertIn(event()['text'],response['text'])
        self.assertAlmostEqual(sum(state['scores'].values()),1)
        self.assertEqual(self.model.parameter_count,29101)

    def test_audio_changes_fusion(self):
        a=event(); b=event(); b['audio_b64']=base64.b64encode(tone(0,99)).decode()
        sa=Session(self.model).process(a)[0]; sb=Session(self.model).process(b)[0]
        self.assertNotEqual(sa['scores'],sb['scores'])

    def test_text_changes_fusion(self):
        a=event(); b=event(); b['text']='This is frustrating'
        self.assertNotEqual(Session(self.model).process(a)[0]['scores'],Session(self.model).process(b)[0]['scores'])

    def test_reject_duplicate_and_recover(self):
        s=Session(self.model); s.process(event())
        with self.assertRaises(ValueError): s.process(event())
        self.assertEqual(s.process(event(1))[0]['seq'],1)

    def test_timestamp_failure_does_not_commit(self):
        s=Session(self.model); bad=event(); bad['end_ms']=2000
        with self.assertRaises(ValueError): s.process(bad)
        s.process(event())

    def test_invalid_inputs(self):
        for key,value in [('audio_b64','bad!'),('text',''),('start_ms',float('nan')),('seq',True)]:
            with self.subTest(key=key):
                e=event(); e[key]=value
                with self.assertRaises(ValueError): Session(self.model).process(e)

    def test_silence_fallback(self):
        out=io.BytesIO()
        with wave.open(out,'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(32000))
        e=event(); e['audio_b64']=base64.b64encode(out.getvalue()).decode()
        s=Session(self.model).process(e)[0]
        self.assertIn('audio_unusable',s['tags']); self.assertEqual(s['evidence']['audio_weight'],0)

    def test_malformed_wav(self):
        with self.assertRaises(ValueError): audio_features(b'not a WAV')
        self.assertEqual(len(audio_features(tone(1,0))[0]),AUDIO_DIM)


if __name__=='__main__': unittest.main()

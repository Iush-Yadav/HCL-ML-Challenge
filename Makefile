PYTHON ?= .venv/bin/python

.PHONY: demo test replay train evaluate

demo:
	$(PYTHON) scripts/demo.py

test:
	$(PYTHON) -m unittest discover -s tests -v

replay:
	$(PYTHON) -m emotion_robot.stream --model models/synthetic.npz < artifacts/demo/input.jsonl

train:
	$(PYTHON) -m emotion_robot.train --data data/meld --output models/meld.npz

evaluate:
	$(PYTHON) -m emotion_robot.evaluate --data data/meld --model models/meld.npz

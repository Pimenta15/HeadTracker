"""Facial landmark detector based on a CNN."""
import os

import cv2
import numpy as np
import onnxruntime as ort


class MarkDetector:
    """Detects 68 facial landmarks from a face crop."""

    def __init__(self, model_file):
        assert os.path.exists(model_file), f"File not found: {model_file}"
        self._input_size = 128
        self.model = ort.InferenceSession(
            model_file, providers=["DmlExecutionProvider", "CPUExecutionProvider"])

    def _preprocess(self, bgrs):
        rgbs = []
        for img in bgrs:
            img = cv2.resize(img, (self._input_size, self._input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)
        return rgbs

    def detect(self, images):
        """Return landmarks as numpy array of shape [Batch, 68*2]."""
        inputs = self._preprocess(images)
        marks = self.model.run(["dense_1"], {"image_input": inputs})
        return np.array(marks)

    def visualize(self, image, marks, color=(255, 255, 255)):
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(mark[1])), 1, color, -1, cv2.LINE_AA)

import unittest
from random import randint

import numpy as np
import torch

from src.agents.NNAgent import NNAgent
from src.utils.onnx_utils import export_to_onnx

import onnxruntime as ort


class MyTestCase(unittest.TestCase):
    batch_size = 1
    state_size = 162
    action_size = 40
    def test_export(self):
        agent = NNAgent((self.state_size,), self.action_size, hidden_size=128)
        agent.load_state_dict(torch.load("agent.pt"))

        export_to_onnx(agent)

    def test_action_mask(self):
        ort_session = ort.InferenceSession("agent.onnx")

        for non_masked_action in range(1, self.action_size+1):
            inputs = np.random.randn(self.batch_size, self.state_size + self.action_size).astype(np.float32)
            inputs[:, -40:] = 0
            inputs[:, -non_masked_action] = 1

            outputs = ort_session.run(
                ["action"],
                {"input": inputs}
            )
            print(outputs)


if __name__ == '__main__':
    unittest.main()

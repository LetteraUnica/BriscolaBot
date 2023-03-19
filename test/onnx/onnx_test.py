import unittest

import numpy as np
import torch

from src.agents.NNAgent import NNAgent
from src.utils.onnx_utils import export_to_onnx

import onnxruntime as ort


class OnnxTest(unittest.TestCase):
    batch_size = 1
    state_size = 162
    action_size = 40
    model_path = "../pretrained_models/briscola-bot-v3.pt"

    def test_action_mask(self):
        agent = NNAgent((self.state_size,), self.action_size, hidden_size=256)
        agent.load_state_dict(torch.load(self.model_path))

        export_to_onnx(agent)

        ort_session = ort.InferenceSession("agent.onnx")

        for non_masked_action in range(1, self.action_size + 1):
            inputs = np.random.randn(self.batch_size, self.state_size + self.action_size).astype(np.float32)
            inputs[:, -40:] = 0
            inputs[:, -non_masked_action] = 1

            outputs = ort_session.run(
                ["action"],
                {"input": inputs}
            )
            self.assertEqual(outputs[0][0], 40 - non_masked_action)


if __name__ == '__main__':
    unittest.main()

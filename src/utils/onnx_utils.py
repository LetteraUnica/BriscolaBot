import numpy as np
import torch

from src.agents import NNAgent


def export_to_onnx(agent: NNAgent, filename="agent.onnx", batch_size=1, output_names=None, input_names=None):
    if output_names is None:
        output_names = ["action"]

    if input_names is None:
        input_names = ["input"]

    dummy_input = torch.randn(batch_size, np.prod(agent.observation_shape) + agent.action_size)
    torch.onnx.export(agent, dummy_input, filename, verbose=False, input_names=input_names, output_names=output_names)
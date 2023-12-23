import fire
import torch

from src.models.model import Net
from src.utils.utils_models import create_onnx_model


def convert_pytorch_to_onnx(
    path_model: str = "mnist_cnn.pt",
    path_onnx: str = "nvidia-triton/\
        model_repository/mnist_cnn/1/model.onnx",
):
    """
    Convert serialized pytorch model to onnx

    Args:
        path_model (str, optional): path to serialized pytorch model.
        Defaults to 'mnist_cnn.pt'.
        path_onnx (str, optional): path for saving onnx model.
        Defaults to 'nvidia-triton/model_repository/mnist_cnn/1/model.onnx'.
    """
    model = Net()
    state_dict = torch.load(path_model)
    model.load_state_dict(state_dict)
    create_onnx_model(path_onnx, model)


if __name__ == "__main__":
    fire.Fire(convert_pytorch_to_onnx)

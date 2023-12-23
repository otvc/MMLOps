import logging
from typing import NoReturn

import numpy as np
import torch
from tritonclient import http
from tritonclient.utils import np_to_triton_dtype

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("dag_creator")
logger.setLevel(logging.INFO)


def client_model_call(
    triton_server,
    model_name: str = "mnist_cnn",
    input_tensor: np.ndarray = None,
    input_name: str = "input",
    output_name: str = "output",
) -> torch.Tensor:
    """
    Send request to triton server model

    Args:
        triton_server (_type_): triton server object
        model_name (str, optional): name of model for request.
                                    Defaults to "mnist_cnn".
        input_tensor (np.ndarray, optional): requested tensors.
        Defaults to np.ones((8, 1, 28, 28), dtype=np.float32).
        input_name (str, optional): name for model input.
                                    Defaults to "input".
        output_name (str, optional): name for model output.
                                     Defaults to "output".

    Returns:
        torch.Tensor: model response.
    """
    inputs, outputs = [], []
    inputs.append(
        http.InferInput(
            input_name,
            input_tensor.shape,
            np_to_triton_dtype(input_tensor.dtype),
        )
    )
    inputs[0].set_data_from_numpy(input_tensor)
    logger.info("Inputs prepared.")

    outputs.append(http.InferRequestedOutput(output_name, binary_data=False))

    results = triton_server.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    logger.info("Results obtained.")

    predictions = results.as_numpy(output_name).argmax(-1)
    print(f"Predicted digits: {predictions}")
    return predictions


def test(url: str = "localhost:8500") -> NoReturn:
    triton_server = http.InferenceServerClient(url)

    test_data = []
    ones_input = np.ones((1, 1, 28, 28), dtype=np.float32)
    for i in range(8):
        test_input = ones_input * i
        test_data.append(test_input)
    test_data = np.concatenate(test_data)

    predictions = client_model_call(triton_server, input_tensor=test_data)
    ground_truth = np.array([6, 6, 5, 5, 5, 5, 5, 5])
    assert np.allclose(predictions, ground_truth)

    triton_server.close()
    logger.info("Client is closed.")


if __name__ == "__main__":
    test()

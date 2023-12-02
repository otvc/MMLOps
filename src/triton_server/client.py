from tritonclient import http
from tritonclient.utils import np_to_triton_dtype
import torch
import numpy as np

def test():
    triton_server = http.InferenceServerClient(url='127.0.0.1:8000')
    model_name = 'add_sub'

    image = np.ones((1, 1, 28, 28), dtype=np.float32)

    inputs, outputs = [], []
    inputs.append(http.InferInput("image", image.shape, np_to_triton_dtype(image.dtype)))
    inputs[0].set_data_from_numpy(image)

    outputs.append(http.InferRequestedOutput("output__0", binary_data=False))

    results = triton_server.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )
    triton_server.close()

    print(results.as_numpy("output__0"))

if __name__ == '__main__':
    test()

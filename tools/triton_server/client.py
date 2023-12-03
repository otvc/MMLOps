import logging

import numpy as np
from tritonclient import http
from tritonclient.utils import np_to_triton_dtype

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("dag_creator")
logger.setLevel(logging.INFO)


def test():
    triton_server = http.InferenceServerClient(url="localhost:8000")
    model_name = "add_sub"
    logger.info("Triton server created.")

    image = np.ones((1, 1, 28, 28), dtype=np.float32)

    inputs, outputs = [], []
    inputs.append(
        http.InferInput("image", image.shape, np_to_triton_dtype(image.dtype))
    )
    inputs[0].set_data_from_numpy(image)
    logger.info("Inputs prepared.")

    outputs.append(http.InferRequestedOutput("output__0", binary_data=False))

    results = triton_server.infer(
        model_name=model_name, inputs=inputs, outputs=outputs
    )
    logger.info("Results obtained.")

    triton_server.close()
    logger.info("Client is closed.")

    print(results.as_numpy("output__0"))


if __name__ == "__main__":
    test()

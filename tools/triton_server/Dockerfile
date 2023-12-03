FROM nvcr.io/nvidia/tritonserver:23.08-pyt-python-py3 as tritonserver

ENV TS_VERSION=23.08

WORKDIR /app/inference_triton

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY model.py model_repository/add_sub/1/model.py
COPY mnist_cnn.pt model_repository/add_sub/1/mnist_cnn.pt
COPY config.pbtxt model_repository/add_sub/config.pbtxt
COPY model.py model_repository/add_sub/1/utils_model.py

CMD tritonserver --model-repository `pwd`/model_repository --backend-config=python,shm-default-byte-size=256777216

FROM nvcr.io/nvidia/tritonserver:23.08-py3-sdk as tritonclient

WORKDIR /app/client

COPY requirements_client.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install tritonclient

ARG address=localhost:8000
ARG model_name=add_sub

COPY client.py client.py

ENV ADDRESS ${address}
ENV MODEL_NAME ${model_name}

CMD python3 client.py

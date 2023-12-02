docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack 67108864 -ti nvcr.io/nvidia/tritonserver:23.09py3
git clone https://github.com/triton-inference-server/python_backend -b r23.09
TRITON_DOCKER_ID=3d0644a0f9cd
docker cp model.py ${TRITON_DOCKER_ID}:/opt/tritonserver/python_backend/models/add_sub/1 
docker cp config.pbtxt ${TRITON_DOCKER_ID}:/opt/tritonserver/python_backend/models/add_sub
docker cp ../../mnist_cnn.pt ${TRITON_DOCKER_ID}:/opt/tritonserver/python_backend/models/add_sub/1
docker cp requirements.txt ${TRITON_DOCKER_ID}:/opt/tritonserver/python_backend

TRITON_CLIENT_DOCKER_ID=d4e28c51a874
docker cp client.py ${TRITON_CLIENT_DOCKER_ID}:/workspace/python_backend/examples/add_sub/
docker cp requirements_client.txt ${TRITON_CLIENT_DOCKER_ID}:/workspace/requirements.txt
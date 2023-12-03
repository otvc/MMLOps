docker-compose build .
docker run --shm-size=7g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 triton_server-tritonserver

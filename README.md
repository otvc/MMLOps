Repository for **MLOps** course in **MIPT**
## Structure
- **train.py** - скрипт для тренировки модели на датасете MNIST
- **infer.py** - скрипт для инференса модели на датасете MNIST
- **test_client.py** - скрипт для теста клиента модели крутящейся на triton inference server;
- **pytorch_to_onnx.py** - скрипт для преобразования сериализованной pytorch модели в onnx формат.

## Triton Inference Server

Инференс модели выполняется при помощи тритон сервера с использованием onnx бэкенда.
Для преобразования сериализованной PyTorch модели в onnx формат необходимо запустить скрипт pytorch_to_onnx.py. По дефолту предполагается, что данный скрипт запускается со следующими параметрами:
```sh
python3 pytorch_to_onnx.py --path_model='mnist_cnn.pt' --path_onnx='nvidia-triton/model_repository/mnist_cnn/1/model.onnx'
```

Перед тем как запускать модель, необходимо подтянуть ее из dvc.
Для этого используйте команду:
```sh
dvc pull nvidia-triton/model_repository/mnist_cnn/1/model.onnx.dvc
```

Для запуска triton server бекенда необходимо перейти в директорию nvidia-triton и выполнить следующую команду:
```sh
docker-compose up
```

Для того, чтобы запустить тест, необходимо выполнить следующую команду:

```sh
python test_client.py
```

# Depricated

В tools/triton_server располагается python_backend для инференса модели.
На данный момент изменен на onnx бэкенд.

Repository for **MLOps** course in **MIPT**
## Structure
- **train.py** - script for training cnn model for MNIST
- **infer.py** - script for inference on test set of MNIST

Инференс модели выполняется при помощи тритон сервера.
Алгоритм:
1. Для того, чтобы поднять инференс, переместите файл с сохраненной после тренировки моделью в `tools/triton_server` с названием `mnist_cnn.pt`.
2. Далее перейдите в папку `tools/triton_server`;
3. Запустите скрипт `deploy.sh`

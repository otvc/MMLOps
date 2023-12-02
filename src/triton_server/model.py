from typing import NoReturn, Dict, Any, List
import sys
from pathlib import Path
import json
from logging import getLogger

logger = getLogger("info_logger")

parrent_dir = Path(__file__).resolve().parent
sys.path.append(str(parrent_dir))
sys.path.append(str(parrent_dir.parent.parent / 'src'))

import torch
from torch import nn
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, image):
        print('Image into model')
        x = self.conv1(image)
        print(x.shape, x)
        print('CONV1')
        x = F.relu(x)
        x = self.conv2(x)
        print('CONV2')
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        print('POOLING1')
        x = self.dropout1(x)
        print('DROPOUT1')
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        print('FC1')
        x = self.dropout2(x)
        x = self.fc2(x)
        print('FC2')
        output = F.log_softmax(x, dim=1)
        print('SOFTMAX1')
        return output

class TritonPythonModel:

    def initialize(self, args: Dict[str, Any]) -> NoReturn:
        config = json.loads(args['model_config'])
        base_params = config['parameters']['base_params']
        path_model = base_params['string_value']
        checkpoint = torch.load(path_model, map_location="cpu")
        self.model = Net()
        self.model.eval()
        # self.model.load_state_dict(checkpoint)
        # self.model = self.model.to('cpu')

    def execute(self, requests: List[Any]) -> List[Any]:
        responses = []
        print(self.model)
        with torch.no_grad():
            for request in requests:
                image = pb_utils.get_input_tensor_by_name(request, 'image')
                image = torch.tensor(image.as_numpy())
                print(image.shape, image)
                image = image.to(next(self.model.parameters()))
                probas = torch.randn((image.shape[0], 10))
                classes = torch.argmax(probas, -1).numpy()
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("output__0", classes)]
                )
                responses.append(inference_response)
        return responses
    

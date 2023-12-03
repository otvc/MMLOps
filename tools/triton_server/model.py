import json
from typing import Any, Dict, List, NoReturn

import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from torch import nn


class Net(nn.Module):
    def __init__(self) -> NoReturn:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, image):
        x = self.conv1(image)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TritonPythonModel:
    def initialize(self, args: Dict[str, Any]) -> NoReturn:
        config = json.loads(args["model_config"])
        base_params = config["parameters"]["base_params"]
        path_model = base_params["string_value"]
        checkpoint = torch.load(path_model, map_location="cpu")
        self.model = Net()
        self.model.eval()
        self.model.load_state_dict(checkpoint)

    def execute(self, requests: List[Any]) -> List[Any]:
        responses = []
        with torch.no_grad():
            for request in requests:
                image = pb_utils.get_input_tensor_by_name(request, "image")
                image = torch.tensor(image.as_numpy())
                image = image.to(next(self.model.parameters()))
                probas = torch.randn((image.shape[0], 10))
                classes = torch.argmax(probas, -1).numpy()
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("output__0", classes)]
                )
                responses.append(inference_response)
        return responses

from __future__ import print_function

from typing import NoReturn

import pandas as pd
import torch
import numpy
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig

from models.train import Net, test


hydra.main(config_path='config', config_name='inference', version_base=False)
def run_inference(config: DictConfig) -> NoReturn:
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(config['seed'])

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_kwargs = {"batch_size": config['batch_size']}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()

    checkpoint = torch.load(config['path_model'])
    model.load_state_dict(checkpoint)
    model = Net().to(device)

    predicts = test(model, device, test_loader)
    df = pd.DataFrame({"Labels": predicts.T[0]})
    df.to_csv(config['path_result'])
    
    input_img = torch.from_numpy(numpy.ones((1, 1, 28, 28), dtype=numpy.float32))
    print(input_img.shape, input_img)
    print(model(input_img))


if __name__ == "__main__":
    run_inference()

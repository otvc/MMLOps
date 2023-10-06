from __future__ import print_function

from typing import NoReturn

import pandas as pd
import torch
from torchvision import datasets, transforms

from train import Net, test


def main(
    batch_size: int = 32,
    no_cuda: bool = False,
    path_model: str = "mnist_cnn.pt",
    path_result: str = "predicts.csv",
    seed: int = 42,
) -> NoReturn:
    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_kwargs = {"batch_size": batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()

    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint)
    model = Net().to(device)

    predicts = test(model, device, test_loader)
    df = pd.DataFrame({"Labels": predicts.T[0]})
    df.to_csv(path_result)


if __name__ == "__main__":
    main()

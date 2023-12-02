from __future__ import print_function

import argparse
from typing import NoReturn, Dict, Any
from pathlib import Path
import sys

cur_parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(cur_parent_dir))
sys.path.insert(1, str(cur_parent_dir.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from utils import utils_data
from model import Net


def train(config: DictConfig,
          model: nn.Module,
          device: str,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          epoch: int) -> NoReturn:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.squeeze(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % config['tracking']['log_interval'] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if config['dry_run']:
                break


def test(model: nn.Module,
         device: str,
         test_loader: DataLoader) -> np.ndarray:
    model.eval()
    test_loss = 0
    correct = 0
    predicts = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target.squeeze(-1), reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predicts.append(pred)
    test_loss /= len(test_loader.dataset)
    predicts = torch.cat(predicts)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return predicts.detach().cpu().numpy()


@hydra.main(config_path="conf", config_name='train', version_base=None)
def run_training(config: DictConfig) -> NoReturn:
    use_cuda = not config['no_cuda'] and torch.cuda.is_available()
    use_mps = not config['no_mps'] and torch.backends.mps.is_available()

    torch.manual_seed(config['seed'])

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_kwargs = {"batch_size": config['hyperparams']['batch_size']}
    test_kwargs = {"batch_size": config['hyperparams']['test_batch_size']}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = utils_data.prepare_train_test_loaderes(
        path_train_split=config['dataset']['path_train_split'],
        path_test_split=config['dataset']['path_test_split'],
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config['hyperparams']['lr'])

    scheduler = StepLR(optimizer, step_size=1, gamma=config['hyperparams']['gamma'])
    for epoch in range(1, config['hyperparams']['epochs'] + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if config['model']['save']:
        torch.save(model.to('cpu').state_dict(), config['model']['path'])


if __name__ == "__main__":
    run_training()

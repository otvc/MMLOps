from __future__ import print_function

from typing import Any, Dict, NoReturn, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.models.model import Net
from src.utils import utils_data


def train(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    loss_history = np.array([])
    ground_truth, predicted = [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output: torch.Tensor = model(data)
        loss = F.nll_loss(output, target.squeeze(-1))
        loss.backward()
        optimizer.step()

        loss_history = np.append(loss_history, loss.detach().cpu().numpy())
        ground_truth.extend(target[:, 0].detach().cpu().numpy())
        pred = output.argmax(dim=1, keepdim=True)
        predicted.extend(pred[:, 0].detach().cpu().numpy())
    avg_train_loss = loss_history.mean()
    clf_report = classification_report(
        ground_truth, predicted, output_dict=True, zero_division=0
    )
    return avg_train_loss, clf_report


def test(
    model: nn.Module, device: str, test_loader: DataLoader
) -> Tuple[np.ndarray, float]:
    test_loss = 0
    correct = 0
    ground_truth, predicts = [], []
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
            ground_truth.append(target)
    test_loss /= len(test_loader.dataset)
    predicts = torch.cat(predicts).detach().cpu().numpy()
    ground_truth = torch.cat(ground_truth).detach().cpu().numpy()
    clf_report = classification_report(
        ground_truth, predicts, output_dict=True, zero_division=0
    )

    return predicts, test_loss, clf_report


def clf_report_into_mlflow(
    clf_report: Dict[str, Any], metric_prefix: str, step: int
) -> NoReturn:
    df_clf_rep = pd.DataFrame(clf_report)
    accuracy = df_clf_rep["accuracy"].iloc[0]
    mlflow.log_metric(f"{metric_prefix}accuracy", accuracy, step=step)
    df_clf_rep.drop(columns=["accuracy"], inplace=True)
    for column in df_clf_rep.columns:
        metric_series = df_clf_rep[column]
        for id_name in metric_series.index:
            metric_value = metric_series[id_name]
            key = f"{metric_prefix}{id_name}"
            mlflow.log_metric(key=key, value=metric_value, step=step)


def train_loop(
    config: Dict[str, Any],
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    epoch: int,
) -> NoReturn:
    mlflow_params = config["logging"]["mlflow_params"]
    exp = mlflow.get_experiment_by_name(name=mlflow_params["name"])
    if not exp:
        exp = mlflow.create_experiment(name=mlflow_params["name"])
    mlflow.set_tracking_uri(uri=mlflow_params["tracking_uri"])
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_params(config)
        for epoch in range(1, config["hyperparams"]["epochs"] + 1):
            train_loss, train_clf_report = train(
                model, device, train_loader, optimizer
            )
            clf_report_into_mlflow(train_clf_report, "train_", epoch)
            _, test_loss, test_clf_report = test(model, device, test_loader)
            clf_report_into_mlflow(test_clf_report, "test_", epoch)
            scheduler.step()
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)


@hydra.main(config_path="conf", config_name="train", version_base=None)
def run_training(config: DictConfig) -> NoReturn:
    use_cuda = not config["no_cuda"] and torch.cuda.is_available()
    use_mps = not config["no_mps"] and torch.backends.mps.is_available()

    torch.manual_seed(config["seed"])

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_kwargs = {"batch_size": config["hyperparams"]["batch_size"]}
    test_kwargs = {"batch_size": config["hyperparams"]["test_batch_size"]}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader, test_loader = utils_data.prepare_train_test_loaderes(
        path_train_split=config["dataset"]["path_train_split"],
        path_test_split=config["dataset"]["path_test_split"],
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(
        model.parameters(), lr=config["hyperparams"]["lr"]
    )
    scheduler = StepLR(
        optimizer, step_size=1, gamma=config["hyperparams"]["gamma"]
    )

    train_loop(
        config=config,
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config["hyperparams"]["epochs"],
    )

    if config["model"]["save"]:
        torch.save(model.to("cpu").state_dict(), config["model"]["path"])


if __name__ == "__main__":
    run_training()

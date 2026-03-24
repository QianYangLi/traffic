import torch


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))


def mape(pred, true):
    return torch.mean(torch.abs((true - pred) / (torch.abs(true) + 1e-8))) * 100
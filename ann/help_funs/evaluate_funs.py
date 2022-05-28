import torch
import numpy as np


def accuracy(model, ds, ok_error=0.2):
    correct = 0
    total = 0
    for val_data, label in ds:
        with torch.no_grad():
            output = model(val_data)
        abs_delta = np.abs(output.item() - label.item())
        if abs_delta < ok_error:
            correct += 1
        total += 1
    acc = correct / total
    return acc

def get_model_guess_vs_target_arr(model, ds):
    model_guess_arr = np.empty(len(ds))
    target_arr = np.empty(len(ds))

    for i, data in enumerate(ds):
        val_data, label = data
        with torch.no_grad():
            output = model(val_data)
        target_arr[i] = label
        model_guess_arr[i] = output

    return target_arr, model_guess_arr

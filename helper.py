import torch


def get_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return num_matches.float()

# This is the metrics.py file in the shoplifting_detection\utils directory
import torch

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(labels)

def precision_recall_fscore(preds, labels):
    tp = torch.sum((preds == 1) & (labels == 1)).float()
    fp = torch.sum((preds == 1) & (labels == 0)).float()
    fn = torch.sum((preds == 0) & (labels == 1)).float()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision.item(), recall.item(), f1.item()
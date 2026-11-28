import torch


def evaluate_clean_accuracy(model, dataloader, device, multi_label=False, threshold=0.68):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if multi_label:
                thresh = threshold if not isinstance(threshold, torch.Tensor) else threshold.to(device)
                predictions = (outputs.sigmoid() > thresh).long()
                correct += (predictions == labels.long()).all(dim=1).sum().item()
            else:
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("The evaluation dataset is empty.")

    return 100.0 * correct / total

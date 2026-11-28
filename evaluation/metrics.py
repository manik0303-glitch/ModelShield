import torch

def evaluate_clean_accuracy(model, dataloader, device):

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

if total == 0:
    raise ValueError("The evaluation dataset is empty.")

return 100.0 * correct / total

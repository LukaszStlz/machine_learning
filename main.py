import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import kagglehub
from model import VegetableNet

if __name__ == '__main__':

    path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(path, transform=transform)

    test_split = 0.2
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(dataset.classes)
    print(num_classes)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    param_grid = [
        {"lr":0.001, "dropout":0.4, "batch_size":32},
        {"lr":0.0005, "dropout":0.3, "batch_size":32},
        {"lr":0.001, "dropout":0.5, "batch_size":16},
    ]

    best_acc = 0
    best_params = None

    for i, params in enumerate(param_grid):
      print(f'Testing params: {params}')
      writer = SummaryWriter(f'runs/experiment_{i}')
      model = VegetableNet(num_classes=num_classes, dropout_rate=params["dropout"]).to(DEVICE)
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=params["lr"])

      train_dl = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
      test_dl = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

      EPOCHS = 5
      for epoch in range(EPOCHS):
          model.train()
          running_loss = 0
          for images, labels in train_dl:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

          avg_loss = running_loss/len(train_dl)
          print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')
          writer.add_scalar('Loss/Train', avg_loss, epoch)

          model.eval()
          correct, total = 0, 0
          val_loss = 0
          with torch.no_grad():
            for images, labels in test_dl:
              images, labels = images.to(DEVICE), labels.to(DEVICE)
              outs = model(images)
              loss = criterion(outs, labels)
              val_loss += loss.item()
              _, predicted = torch.max(outs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

          acc = 100 * correct / total
          avg_val_loss = val_loss / len(test_dl)
          print(f'Validation Acc: {acc:.2f}%')
          writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
          writer.add_scalar('Accuracy/Validation', acc, epoch)

          if acc > best_acc:
            best_acc = acc
            best_params = params

      writer.add_hparams(params, {'accuracy': best_acc})
      writer.close()
      print(f'Best accuracy for this run: {acc:.2f}%')

    print(f'Overall best accuracy: {best_acc:.2f}% with params: {best_params}')




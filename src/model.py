import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from helpers.utils import LaughterDataset
from sklearn.model_selection import train_test_split


class LaughCNN(nn.Module):
  def __init__(self):
    super(LaughCNN, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32 * 3 * 32, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
  )

  def forward(self, x):
    x = self.conv(x)
    return self.fc(x)


def train_model(model, train_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(20):
    model.train()
    running_loss = 0.0
    for audio, _, _, numerical in train_loader:
        outputs = model(audio)
        loss = criterion(outputs, numerical)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")


def test_model(model, test_loader):
  model.eval()
  preds, true_labels = [], []

  with torch.no_grad():
      for audio, audio_id, _, numerical in test_loader:
          outputs = model(audio)
          _, predicted = torch.max(outputs, 1)
          preds.extend(predicted.cpu().numpy())
          true_labels.extend(numerical.numpy())
          print('Predicted: ', predicted)
          print('True: ', numerical)


def main():
  labels = pd.read_csv('data/labels-2.csv')
  mappings = {'belly': 0, 'chuckle': 1, 'baby': 2} 
  labels['laugh_numerical'] = labels['laugh_class'].map(mappings)

  ROOT_PATH = Path('data/training-dataset-2')

  # Split between training (90%) and testing (10%) labels 
  training_labels, testing_labels = train_test_split(labels, train_size=0.9, test_size=0.1)

  # This will return datasets splited between training and labels
  training_dataset = LaughterDataset(ROOT_PATH, labels=training_labels)
  testing_dataset = LaughterDataset(ROOT_PATH, labels=testing_labels)

  # Creating Dataloaders for Neural Network
  train_loader = DataLoader(training_dataset, batch_size=8, shuffle=True)
  test_loader = DataLoader(testing_dataset, batch_size=8, shuffle=False)

  model = LaughCNN()

  train_model(model, train_loader)
  test_model(model, test_loader)

main()
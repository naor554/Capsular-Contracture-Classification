import os
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet18
import torch.nn as nn
import torch.optim as optim
import csv
import random

def npy_loader(file_path):
    return np.load(file_path)

train_transforms = Compose([
    # custom or adapted transforms for .npy loading
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 128)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 128)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

def main():
    csv_file = "data_paths_labels.csv"  # your CSV file

    all_items = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)  # expects columns: image_path, label
        for row in reader:
            image_path = row["image_path"]
            label = row["label"]
            all_items.append({"image": image_path, "label": label})

    # Shuffle the entire dataset
    random.shuffle(all_items)

    # Define split ratios
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate sizes
    data_size = len(all_items)
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)

    # Slice dataset
    train_list = all_items[:train_size]
    val_list = all_items[train_size: train_size + val_size]

    print("Train:", len(train_list), "Validation:", len(val_list))

    train_ds = Dataset(data=train_list, transform=train_transforms)
    val_ds = Dataset(data=val_list, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    max_epochs = 10
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}\n")

if __name__ == "__main__":
    main()

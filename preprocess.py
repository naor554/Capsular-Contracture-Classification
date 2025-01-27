import os
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord,LoadImaged
from monai.data import Dataset, DataLoader, NumpyReader
from monai.networks.nets import resnet18
import torch.nn as nn
import torch.optim as optim
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


train_transforms = Compose([
    LoadImaged(keys=["image"], reader=NumpyReader),
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 32)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image"], reader=NumpyReader),
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 32)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

def main():
    csv_file = "data_paths_labels.csv"  # your CSV file

    all_items = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)  # expects columns: image, label
        for row in reader:
            image_path = row["image"]
            label = int(row["label"])
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

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
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

    ###############################################################################
    # 5) Final Evaluation: Confusion Matrix, Metrics, and Plots
    ###############################################################################
    # We'll re-run inference on the validation set to collect predictions & probabilities
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            outputs = model(val_images)  # shape [B, 2]
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class "1"

            _, predicted_labels = torch.max(outputs, dim=1)

            y_true.extend(val_labels.cpu().numpy().tolist())
            y_pred.extend(predicted_labels.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # For ROC-AUC, we need probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n========== Final Evaluation on Validation Set ==========")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")

    # Plot the confusion matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Plot the ROC curve
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()

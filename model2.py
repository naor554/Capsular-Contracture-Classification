import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, ResizeD, ToTensord, LoadImaged, RandFlipd, \
    RandRotated, RandGaussianNoised
from monai.data import Dataset, DataLoader, NumpyReader
from monai.networks.nets import resnet18
import torch.nn as nn
import torch.optim as optim
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    confusion_matrix

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations with augmentations
train_transforms = Compose([
    LoadImaged(keys=["image"], reader=NumpyReader),
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 32)),
    ScaleIntensityd(keys=["image"]),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandRotated(keys=["image"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image"], reader=NumpyReader),
    EnsureChannelFirstd(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=(128, 128, 32)),
    ScaleIntensityd(keys=["image"]),
    ToTensord(keys=["image", "label"]),
])

# Load dataset
csv_file = "data_labels.csv"
all_items = []
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        image_path = row["image"]
        label = int(row["label"])
        all_items.append({"image": image_path, "label": label})

random.shuffle(all_items)

# Define K-Fold Cross Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define model outside K-Fold loop
model = resnet18(spatial_dims=2, n_input_channels=1, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store final metrics
final_metrics = {"train_loss": [], "val_loss": [], "val_accuracy": [], "precision": [], "recall": [], "f1_score": [],
                 "roc_auc": []}
all_y_true, all_y_pred, all_y_prob = [], [], []
fold_train_losses, fold_val_losses = [], []

# Initialize best model tracking
best_accuracy = 0.0
best_model_path = "best_model.pth"

# Perform K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(all_items)):
    print(f"\n🔹 Starting Fold {fold + 1}/{k}")

    train_subset = [all_items[i] for i in train_idx]
    val_subset = [all_items[i] for i in val_idx]

    train_ds = Dataset(train_subset, transform=train_transforms)
    val_ds = Dataset(val_subset, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Training and validation loop
    num_epochs = 10
    fold_train_loss, fold_val_loss = [], []
    y_true_fold, y_pred_fold, y_prob_fold = [], [], []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        fold_train_loss.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation after every epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                y_true_fold.extend(labels.cpu().numpy())
                y_pred_fold.extend(preds.cpu().numpy())
                y_prob_fold.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        fold_val_loss.append(avg_val_loss)
        print(f"Val Loss: {avg_val_loss:.4f}")

    fold_train_losses.append(fold_train_loss)
    fold_val_losses.append(fold_val_loss)
    final_metrics["train_loss"].append(np.mean(fold_train_loss))
    final_metrics["val_loss"].append(np.mean(fold_val_loss))
    fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

    # Save best model
    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        best_model_path = f"best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated: Fold {fold + 1} with Accuracy {best_accuracy:.4f}")

    all_y_true.extend(y_true_fold)
    all_y_pred.extend(y_pred_fold)
    all_y_prob.extend(y_prob_fold)

# Compute final evaluation metrics
cm = confusion_matrix(all_y_true, all_y_pred)
accuracy = accuracy_score(all_y_true, all_y_pred)
precision = precision_score(all_y_true, all_y_pred)
recall = recall_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred)
roc_auc = roc_auc_score(all_y_true, all_y_prob)

# Print metrics
print("\n========== Final Evaluation ==========")
print("Confusion Matrix:\n", cm)
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")


# Plot train and validation loss for each fold
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(range(1, len(fold_train_losses[i]) + 1), fold_train_losses[i], linestyle='--', label=f"Fold {i+1} Train Loss")
    plt.plot(range(1, len(fold_val_losses[i]) + 1), fold_val_losses[i], label=f"Fold {i+1} Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss per Fold")
plt.legend()
plt.show()


# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()




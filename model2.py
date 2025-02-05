import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ResizeD,
    ToTensord,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandGaussianNoised
)
from monai.data import Dataset, DataLoader, NumpyReader
from monai.networks.nets import resnet18
import torch.nn as nn
import torch.optim as optim
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for training (with augmentations) and validation
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

# Load dataset from CSV file
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

# Containers for overall metrics and loss curves
final_metrics = {"train_loss": [], "val_loss": [], "val_accuracy": []}
all_y_true, all_y_pred, all_y_prob = [], [], []
fold_train_losses, fold_val_losses = [], []

best_accuracy = 0.0  # Best fold accuracy across folds
best_model_path = "best_model.pth"
num_epochs = 10

# Begin K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(all_items)):
    print(f"\n🔹 Starting Fold {fold + 1}/{k}")

    # Split data for current fold
    train_subset = [all_items[i] for i in train_idx]
    val_subset = [all_items[i] for i in val_idx]

    # Create MONAI datasets and dataloaders
    train_ds = Dataset(data=train_subset, transform=train_transforms)
    val_ds = Dataset(data=val_subset, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Reinitialize model and optimizer for this fold
    model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    fold_train_loss = []
    fold_val_loss = []

    # Training loop for current fold
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        fold_train_loss.append(avg_train_loss)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}")

        # Validation loss per epoch (without prediction collection)
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        fold_val_loss.append(avg_val_loss)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}: Val Loss: {avg_val_loss:.4f}")

    # After training, perform a final evaluation on the validation set
    model.eval()
    y_true_fold, y_pred_fold, y_prob_fold = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            y_true_fold.extend(labels.cpu().numpy())
            y_pred_fold.extend(preds.cpu().numpy())
            y_prob_fold.extend(probs.cpu().numpy())

    fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
    print(f"Fold {fold + 1} Final Accuracy: {fold_accuracy:.4f}")

    # Save the model if it is the best so far (across folds)
    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        best_model_path = f"best_model_fold{fold + 1}.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated: Fold {fold + 1} with Accuracy {best_accuracy:.4f}")

    # Save per-fold metrics and loss curves
    final_metrics["train_loss"].append(np.mean(fold_train_loss))
    final_metrics["val_loss"].append(np.mean(fold_val_loss))
    final_metrics["val_accuracy"].append(fold_accuracy)
    fold_train_losses.append(fold_train_loss)
    fold_val_losses.append(fold_val_loss)

    # Aggregate validation predictions from all folds for overall evaluation
    all_y_true.extend(y_true_fold)
    all_y_pred.extend(y_pred_fold)
    all_y_prob.extend(y_prob_fold)

# Compute overall evaluation metrics using aggregated predictions
cm = confusion_matrix(all_y_true, all_y_pred)
overall_accuracy = accuracy_score(all_y_true, all_y_pred)
precision = precision_score(all_y_true, all_y_pred)
recall = recall_score(all_y_true, all_y_pred)
f1 = f1_score(all_y_true, all_y_pred)
roc_auc = roc_auc_score(all_y_true, all_y_prob)

print("\n========== Final Evaluation ==========")
print("Confusion Matrix:\n", cm)
print(f"Accuracy:    {overall_accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")

# Plot training loss for each fold
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(range(1, num_epochs + 1), fold_train_losses[i], linestyle='--', label=f"Fold {i + 1}")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.title("Train Loss per Fold")
plt.legend()
plt.show()

# Plot validation loss for each fold
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.plot(range(1, num_epochs + 1), fold_val_losses[i], label=f"Fold {i + 1}")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss per Fold")
plt.legend()
plt.show()

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# comenzaremos con nuestra cnn desde cero

# %%
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Device:", device)

#limpiar memoria de GPU
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
from torchvision import datasets, transforms

train_transforms = transforms.Compose([
    transforms.Resize((150,200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],  # media ImageNet
                         [0.229,0.224,0.225])  # std ImageNet
])

val_transforms = transforms.Compose([
    transforms.Resize((150,200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# Ruta a tu dataset en Drive (ajusta según tu estructura)
# dataset_dir = "/content/drive/MyDrive/dataset_lab_2_IA2/"
dataset_dir = "dataset/"

# %%
from torch.utils.data import DataLoader, random_split

full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)

# split 80/20
train_size = int(0.8*len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}
classes = full_dataset.classes
print("Clases:", classes)


# %%
import torchvision.models as models
import torch.nn as nn

model = models.convnext_tiny(weights="IMAGENET1K_V1")  # preentrenado
num_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_features, len(classes))  # reemplaza FC final

model = model.to(device)



# %%
import numpy as np
from tqdm import tqdm

def fit(model, dataloaders, device, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    # Listas para graficar
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    all_labels, all_preds = [], []

    for epoch in range(1, epochs+1):
        # --- TRAIN ---
        model.train()
        train_loss, train_acc = [], []
        for X, y in tqdm(dataloaders['train'], desc=f"Epoch {epoch} Train"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, dim=1)).sum().item() / len(y)
            train_acc.append(acc)
        train_losses.append(np.mean(train_loss))
        train_accs.append(np.mean(train_acc))

        # --- VAL ---
        model.eval()
        val_loss, val_acc = [], []
        epoch_labels, epoch_preds = [], []
        with torch.no_grad():
            for X, y in tqdm(dataloaders['val'], desc=f"Epoch {epoch} Val"):
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, dim=1)).sum().item() / len(y)
                val_acc.append(acc)

                epoch_labels.extend(y.cpu().numpy())
                epoch_preds.extend(torch.argmax(y_hat, dim=1).cpu().numpy())

        val_losses.append(np.mean(val_loss))
        val_accs.append(np.mean(val_acc))
        all_labels.append(epoch_labels)
        all_preds.append(epoch_preds)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.5f}, Train Acc: {train_accs[-1]:.5f} | "
              f"Val Loss: {val_losses[-1]:.5f}, Val Acc: {val_accs[-1]:.5f}")

    # Devolver listas y predicciones para gráficos y matriz de confusión
    return val_loss, train_acc, val_losses, val_accs, all_labels, all_preds


# %%
train_losses, train_accs, val_losses, val_accs, all_labels, all_preds = fit(
    model, 
    dataloaders, 
    device, 
    epochs=15, 
    lr=1e-4
)

# %%
import matplotlib.pyplot as plt

epochs_range = range(1, len(train_losses)+1)

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss por Epoch")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(epochs_range, train_accs, label="Train Acc")
plt.plot(epochs_range, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy por Epoch")
plt.legend()

plt.show()

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Tomamos las últimas predicciones del último epoch
y_true = all_labels[-1]
y_pred = all_preds[-1]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión - Último Epoch")
plt.show()



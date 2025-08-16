# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# %%
# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)

# %%
# --- TRANSFORMACIONES ---
train_transforms = transforms.Compose([
    transforms.Resize((300, 400)),  # mitad del tamaño original
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

val_transforms = transforms.Compose([
    transforms.Resize((300, 400)),  # mitad del tamaño original
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])


# %%
# --- DATASET ---
dataset_dir = "./dataset"
full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transforms)

# Split 80% train / 20% validación
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
classes = full_dataset.classes

# %%

# Cambiar transformaciones de validación
val_dataset.dataset.transform = val_transforms

# %%

# --- DATALOADERS ---
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)  # batch pequeño por VRAM
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

dataloader = {'train': train_loader, 'test': val_loader}

# %%
def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

import torchvision
imshow(torchvision.utils.make_grid(images))
print("Etiquetas:", [classes[labels[j]] for j in range(len(labels))])


# %%
# --- BLOQUES ---
def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        nn.ReLU(),
        nn.MaxPool2d(pk, stride=ps)
    )

# %%
# --- MODELO ---
class CNN(nn.Module):
    def __init__(self, n_channels=3, n_outputs=5):
        super().__init__()
        self.conv1 = block(n_channels, 64)   # 300x400 -> 150x200
        self.conv2 = block(64, 128)          # 150x200 -> 75x100
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10,10))  # salida 128x10x10
        self.fc = nn.Linear(128*10*10, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

model = CNN(n_channels=3, n_outputs=5).to(device)
print(model)

# %%
# --- FUNCION DE ENTRENAMIENTO CON TRACKING ---
def fit_track(model, dataloader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Listas para graficar
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    all_labels, all_preds = [], []

    for epoch in range(1, epochs+1):
        # --- ENTRENAMIENTO ---
        model.train()
        t_loss, t_acc = [], []
        for X, y in tqdm(dataloader['train'], desc=f"Epoch {epoch} Train"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            t_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, dim=1)).sum().item() / len(y)
            t_acc.append(acc)
        train_losses.append(np.mean(t_loss))
        train_accs.append(np.mean(t_acc))

        # --- VALIDACIÓN ---
        model.eval()
        v_loss, v_acc = [], []
        epoch_labels, epoch_preds = [], []
        with torch.no_grad():
            for X, y in tqdm(dataloader['test'], desc=f"Epoch {epoch} Val"):
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                v_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, dim=1)).sum().item() / len(y)
                v_acc.append(acc)

                # Guardar predicciones y etiquetas
                epoch_labels.extend(y.cpu().numpy())
                epoch_preds.extend(torch.argmax(y_hat, dim=1).cpu().numpy())

        val_losses.append(np.mean(v_loss))
        val_accs.append(np.mean(v_acc))
        all_labels.append(epoch_labels)
        all_preds.append(epoch_preds)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.5f}, Train Acc: {train_accs[-1]:.5f} | "
              f"Val Loss: {val_losses[-1]:.5f}, Val Acc: {val_accs[-1]:.5f}")

    # Devolver listas y predicciones para gráficos y matriz de confusión
    return train_losses, train_accs, val_losses, val_accs, all_labels, all_preds


# %%
# --- ENTRENAR MODELO ---
train_losses, train_accs, val_losses, val_accs, all_labels, all_preds = fit_track(
    model, 
    dataloader, 
    device, 
    epochs=8, 
    lr=1e-3
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



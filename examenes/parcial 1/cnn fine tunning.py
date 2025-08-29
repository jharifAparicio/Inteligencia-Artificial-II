# %%
import torch
import torchvision
import torchvision.transforms as transforms
# import albumentations as A
# from skimage import io
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%

# --------------------------
# Modelo con MNASNet
# --------------------------
class MnasNetCustom(torch.nn.Module):
    def __init__(self, n_outputs=10, pretrained=False, freeze=False):
        super().__init__()
        mnasnet = torchvision.models.mnasnet0_5(weights="IMAGENET1K_V1" if pretrained else None)

        # Feature extractor
        self.features = mnasnet.layers

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # Clasificador nuevo
        self.fc = torch.nn.Linear(mnasnet.classifier[1].in_features, n_outputs)

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # igual que en el modelo original
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

# %%
# --------------------------
# Loop de entrenamiento
# --------------------------
def fit(model, dataloader, epochs=5, lr=1e-2):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'])
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc)
            bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")

        # Validación
        bar = tqdm(dataloader['test'])
        val_loss, val_acc = [], []
        model.eval()
        with torch.no_grad():
            for X, y in bar:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")

        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")

# %%
# --------------------------
# MNIST listo para MNASNet
# --------------------------
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision

# Transformaciones correctas para MNASNet
transform = transforms.Compose([
    transforms.Resize(128),                        # MNASNet necesita imagen más grande
    transforms.Grayscale(num_output_channels=3),   # Convertir 1 canal a 3
    transforms.ToTensor(),
    transforms.Normalize(                          # Normalización tipo ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Datasets
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True,
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True,
    transform=transform
)

# DataLoaders
dataloader = {
    "train": DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True),
    "test": DataLoader(test_dataset, batch_size=256, shuffle=False)
}


# %%
# Modelo con base congelada
model = MnasNetCustom(n_outputs=10, pretrained=True, freeze=True)
model.to(device)

# Entrenar solo la capa final
fit(model, dataloader, epochs=20, lr=1e-3)


# %%
model.unfreeze()

fit(model, dataloader, epochs=10, lr=1e-4)

# %%
model.unfreeze()

fit(model, dataloader, epochs=10, lr=1e-4)

# %%
model.unfreeze()

fit(model, dataloader, epochs=10, lr=5e-5)

# %%
torch.save(model.state_dict(), 'modelo_actas.pth')

# %%
# cargamos el modelo, que guardamos
# Crear el modelo
model_path = 'modelo_actas.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MnasNetCustom(n_outputs=10, pretrained=False, freeze=False)
        
# Cargar pesos
checkpoint = torch.load(model_path, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
    
model = model.to(DEVICE)
model.eval()
print(f"✅ Modelo MnasNetCustom cargado desde {model_path}")

# %%
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. Evaluación en test ---
y_true, y_pred = [], []
test_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for X, y in dataloader['test']:
        X, y = X.to(device), y.to(device)   # <- mover batch al device
        outputs = model(X)
        loss = criterion(outputs, y)
        test_loss += loss.item()

        preds = torch.argmax(outputs, 1)
        y_true.extend(y.cpu().numpy())      # <- devolver a CPU para métricas
        y_pred.extend(preds.cpu().numpy())
        
# --- 3. Métricas ---
avg_loss = test_loss / len(dataloader['test'])
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Error en test (loss): {avg_loss:.4f}")
print(f"Precisión (accuracy): {accuracy:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred))

# --- 4. Graficar matriz de confusión ---
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()



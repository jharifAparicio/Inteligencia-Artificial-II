# %%
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# %% [markdown]
# # 2. Cargar y explorar el dataset

# %%
# Cargar archivo CSV
df = pd.read_csv('spam.csv', encoding='latin-1')  # Ajusta el encoding si es necesario
df = df[['v1', 'v2']]  # Asegúrate que las columnas se llamen así (v1: label, v2: texto)
df.columns = ['label', 'text']

print(df.head())
print(df['label'].value_counts())


# %% [markdown]
# # 3. preprocesamiento

# %%
# Codificar etiquetas
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # spam = 1, ham = 0

# Vectorizar texto
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

# Separar en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# %% [markdown]
# # 6. definimos el modelo que usaremos MLP

# %%
class SpamMLP(nn.Module):
    def __init__(self, input_dim):
        super(SpamMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = SpamMLP(input_dim=X_train.shape[1])
model.to(device)


# %% [markdown]
# # entrenamiento

# %%
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
batch_size = 64

losses = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])

    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices].view(-1, 1)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# %% [markdown]
# # evaluacion

# %%
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    preds = (preds > 0.5).float().cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

print("Accuracy:", accuracy_score(y_true, preds))
print(classification_report(y_true, preds))

# %% [markdown]
# # visualizacion de perdida

# %%
plt.plot(losses)
plt.title("Curva de pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()




# %%
# =============================
# 1️⃣ Librerías
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = "cpu"


# %%
# =============================
# 2️⃣ Cargar y preparar datos
# =============================
df = pd.read_csv("Canadian_climate_history.csv")  # Ajusta según tu archivo
df['LOCAL_DATE'] = pd.to_datetime(df['LOCAL_DATE'])
df = df.sort_values('LOCAL_DATE')

# Elegimos Calgary
series = df['MEAN_TEMPERATURE_CALGARY'].dropna().values.reshape(-1,1)

# Normalizar
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Crear secuencias y multi-step 7 días
seq_length =30
pred_steps = 2
X, y = [], []

for i in range(len(series_scaled)-seq_length-pred_steps):
    X.append(series_scaled[i:i+seq_length])
    y.append(series_scaled[i+seq_length:i+seq_length+pred_steps])

X = np.array(X)
y = np.array(y)  # forma (num_samples, 7, 1)

# %%
# =============================
# 3️⃣ Split train/valid/test
# =============================
train_size = int(0.7*len(X))
valid_size = int(0.15*len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_valid, y_valid = X[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
X_test, y_test = X[train_size+valid_size:], y[train_size+valid_size:]

# %%
# =============================
# 4️⃣ Dataset y DataLoader
# =============================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.train:
            return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]).float()
        return torch.from_numpy(self.X[idx]).float()

dataset = {
    'train': TimeSeriesDataset(X_train, y_train),
    'eval': TimeSeriesDataset(X_valid, y_valid),
    'test': TimeSeriesDataset(X_test, y_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], batch_size=64, shuffle=True),
    'eval': DataLoader(dataset['eval'], batch_size=64, shuffle=False),
    'test': DataLoader(dataset['test'], batch_size=64, shuffle=False)
}


# %%
# =============================
# 5️⃣ Modelo LSTM
# =============================
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_steps=2):
        super().__init__()
        self.pred_steps = pred_steps
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # shape: (batch_size, pred_steps)
        return out.unsqueeze(-1)      # agrega dimensión final → (batch_size, pred_steps, 1)


model = LSTMModel().to(device)


# %%
# =============================
# 6️⃣ Función de entrenamiento
# =============================
def fit(model, dataloader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    
    for epoch in bar:
        model.train()
        train_loss = []
        for X_batch, y_batch in dataloader['train']:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        model.eval()
        eval_loss = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader['eval']:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_hat = model(X_batch)
                eval_loss.append(criterion(y_hat, y_batch).item())
                
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {np.mean(eval_loss):.5f}")


# %%
fit(model, dataloader, epochs=50)


# %%
# =============================
# 7️⃣ Predicción
# =============================
def predict(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch in dataloader:  # iteramos directamente sobre el DataLoader
            if isinstance(X_batch, (list, tuple)):
                X_batch = X_batch[0]  # ignoramos y si viene
            X_batch = X_batch.to(device)
            y_hat = model(X_batch)
            preds.append(y_hat.cpu().numpy())
    return np.concatenate(preds, axis=0)


y_pred = predict(model, dataloader['test'])

# %%
# =============================
# 8️⃣ Desnormalizar y evaluar
# =============================
# Para multi-step, desnormalizamos cada día
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1)).reshape(y_test.shape)
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)

mse = mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# %%
# =============================
# 9️⃣ Visualización 6 ejemplos (2x3)
# =============================
def plot_grid_examples(X, y_true, y_pred, n_rows=2, n_cols=3):
    plt.figure(figsize=(20, 10))
    
    for i in range(n_rows * n_cols):
        if i >= len(X):
            break
        plt.subplot(n_rows, n_cols, i+1)
        seq_len = X.shape[1]
        # Secuencia de entrada
        plt.plot(range(seq_len), X[i].flatten(), ".-", label="Input sequence")
        # Valores reales siguientes
        plt.plot(range(seq_len, seq_len+y_true.shape[1]), y_true[i], "bx", markersize=10, label="Real next")
        # Predicción
        plt.plot(range(seq_len, seq_len+y_pred.shape[1]), y_pred[i], "ro", label="Predicted next")
        plt.title(f"Ejemplo {i+1}")
        plt.grid(True)
        if i % n_cols == 0:
            plt.ylabel("Temperatura (°C)")
        if i >= n_rows * n_cols - 1:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_grid_examples(X_test, y_test_rescaled, y_pred_rescaled, n_rows=2, n_cols=3)




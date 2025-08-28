# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import torch
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# %%
# hiperparamentros
batch_size = 64

# %%
import torchvision
from torchvision import transforms

# Transformaciones para MNASNet
transform = transforms.Compose([
    transforms.Resize(224),                  # Redimensiona a 224x224
    transforms.Grayscale(num_output_channels=3),  # Convierte 1 canal a 3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Mean y std de ImageNet
                         std=[0.229, 0.224, 0.225]),
    # data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
])

trainset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# %%
# creamos el data loader para dividir en baches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# %%
# Tomamos un batch
imgs, labels = next(iter(trainloader))

# Seleccionamos las primeras 10
sample_imgs = imgs[:10]
sample_labels = labels[:10]

print(sample_imgs.shape)  # torch.Size([10, 3, 224, 224])
print(sample_labels)

# %%
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# nombres de las clases
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# Convertir tensor a imagen para mostrar
def imshow(img):
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  # Desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Mostrar 10 imágenes
for i in range(5):
    imshow(sample_imgs[i])
    # plt.title(sample_labels[i].item())
    plt.title(classes[sample_labels[i].item()])
    plt.show()


# %%
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size=100):
        super().__init__()
        self.input_size = input_size

        self.inp = nn.Sequential(
            nn.Linear(self.input_size, 14*14*128),
            nn.BatchNorm1d(14*14*128),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),   # 28x28 -> 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),   # 56x56 -> 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1, bias=False),    # 112x112 -> 224x224
            nn.Tanh()  # Salida [-1, 1]
        )

    def forward(self, x):
        x = self.inp(x)
        x = x.view(-1, 128, 14, 14)
        x = self.main(x)
        return x

# mostramos la arquitecura
generator = Generator()
print(generator)

# %%
import torch
import torch.nn as nn
from torchvision.models import mnasnet0_5

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        mnasnet = mnasnet0_5(pretrained=True)
        # Tomar todo excepto el clasificador final
        self.feature_extractor = nn.Sequential(*list(mnasnet.children())[:-1])
        # Linear sobre 1280 canales
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.3), # apagamos un 30% de neuronas aleatoriamente
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.feature_extractor(x)        # (batch, 1280, H, W)
        x = x.mean([2, 3])                   # Global Average Pooling → (batch, 1280)
        x = self.classifier(x)               # (batch, 1)
        return x


# %%
# batch de prueba: 8 imágenes, 3 canales, 224x224
test_input = torch.randn(8, 3, 224, 224)

discriminator = Discriminator()
output = discriminator(test_input)

print(output.shape)  # torch.Size([64,1])
# print()


# %%
# helper para normalizar como ImageNet
imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def normalize_generated(imgs):
    # imgs viene en [-1,1], lo llevamos a [0,1] y luego normalizamos como ImageNet
    imgs = (imgs + 1) / 2
    return imagenet_norm(imgs)

# %%
import matplotlib.pyplot as plt
import torchvision

def show_generated_image(gen, device, noise_size, normalize=True):
    # Generamos una imagen de ejemplo
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(1, noise_size).to(device)
        img = gen(noise)
        if normalize:
            img = (img + 1) / 2  # de [-1,1] a [0,1]

    img = img.squeeze(0).cpu()  # (3,H,W)
    img = torchvision.utils.make_grid(img)  # si hay batch o multiple canales
    plt.imshow(img.permute(1,2,0))
    plt.axis('off')
    plt.show()

# %%
from fastprogress import master_bar, progress_bar

def fit(g, d, dataloader, crit=None, epochs=30):
  g.to(device)
  d.to(device)
  # -------------------------------
  # Optimizadores para la GAN
  # -------------------------------

  # Optimizador del Generador (G)
  # - Usamos Adam, un optimizador adaptativo muy común en GANs.
  # - lr=2e-4: tasa de aprendizaje relativamente alta para que G pueda aprender rápido.
  g_optimizer = torch.optim.Adam(g.parameters(), lr=1e-4)


  # Optimizador del Discriminador (D)
  # - También usamos Adam.
  # - lr=1e-4: un poco más bajo que G para evitar que D aprenda demasiado rápido y domine al generador.
  d_optimizer = torch.optim.Adam(d.parameters(), lr=1e-4)

  crit = nn.BCEWithLogitsLoss() if crit == None else crit
  g_loss, d_loss = [], []
  mb = master_bar(range(1, epochs+1))
  hist = {'g_loss': [], 'd_loss': []}
  for epoch in mb:
    for X, y in progress_bar(trainloader, parent=mb):
      X = X.to(device)
      # --- Entrenamineto del discrimnador ---
      g.eval()
      d.eval()

      #   generamos un batch de imágenes falsas
      noise = torch.randn(X.size(0), g.input_size).to(device)
      generated_images = g(noise)  # (batch, 3, 224, 224)
      generated_images = normalize_generated(generated_images)

      #   input del discrminator -> esto ya no se usa porque nosecitamos la forma espacial de las imagenes
      # d_input = torch.cat([genenerated_images, X.view(X.size(0), -1)])

      # Salidas del discriminador
      d_real = d(X)
      d_fake = d(generated_images)

      # Ground truth
      d_real_gt = torch.ones(X.size(0),1).to(device) # -> esto confunde al discriminador
      # d_real_gt = torch.zeros((X.size(0),1), 0.9 ,device=device)
      d_fake_gt = torch.zeros(X.size(0),1).to(device)

      # Pérdida y optimización
      d_optimizer.zero_grad()
      d_loss_real = crit(d_real, d_real_gt)
      d_loss_fake = crit(d_fake, d_fake_gt)
      d_l = (d_loss_real + d_loss_fake) / 2
      d_l.backward()
      d_optimizer.step()
      d_loss.append(d_l.item())

      # --- Entrenamiento del generador ---
      g.train()
      d.eval() # congela las estadisticas del BN del D si las tiene

      noise = torch.randn(X.size(0), g.input_size).to(device)
      gen_images = g(noise)
      gen_images = normalize_generated(gen_images)

      d_output = d(gen_images)
      g_gt = torch.ones(X.size(0),1).to(device)  # queremos engañar al discriminador

      g_optimizer.zero_grad()
      g_l = crit(d_output, g_gt)
      g_l.backward()
      g_optimizer.step()
      g_loss.append(g_l.item())

      # Log
      mb.child.comment = f'g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}'

    mb.write(f'Epoch {epoch}/{epochs} g_loss {np.mean(g_loss):.5f} d_loss {np.mean(d_loss):.5f}')

    if epoch % 10 == 0:  # cada 10 épocas, por ejemplo
      show_generated_image(g, device, g.input_size)

    hist['g_loss'].append(np.mean(g_loss))
    hist['d_loss'].append(np.mean(d_loss))
  return hist

# %%
# entrenemos
hist = fit(generator, discriminator, trainloader, crit=torch.nn.BCELoss())

# %%
#guardamos los pesos del generador y el discriminador
torch.save(generator.state_dict(), '/content/drive/MyDrive/USFX/INTELIGENCIA ARTIFICIAL II/practicas/GAN/generator_1.pth')
torch.save(discriminator.state_dict(), '/content/drive/MyDrive/USFX/INTELIGENCIA ARTIFICIAL II/practicas/GAN/discriminator_1.pth')

# %%
import pandas as pd

# Convertir el diccionario en un DataFrame
df = pd.DataFrame(hist)
df.plot(grid=True)
plt.show()

# %%
#prueba del generador
generator.eval()
with torch.no_grad():
    noise = torch.randn((64, generator.input_size)).to(device)
    generated_images = generator(noise)  # (10, 3, 224, 224)

    fig, axs = plt.subplots(2,8,figsize=(21,7))
    i = 0
    for ax in axs:
        for _ax in ax:
            img = generated_images[i].permute(1,2,0).cpu()  # C,H,W → H,W,C
            img = (img + 1)/2  # si normalizaste entre -1 y 1
            _ax.imshow(img)
            _ax.axis('off')
            i += 1
    plt.show()



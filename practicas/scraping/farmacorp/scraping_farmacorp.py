import os
import time
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Crear carpeta para guardar imágenes
os.makedirs("imagenes_farmacorp", exist_ok=True)

# Inicializar navegador
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Opcional: sin abrir navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 1. Scraping URLs de categorías
url_principal = "https://farmacorp.com/collections?page="

# num_paginas = 30

urls_categorias = []

for i in range(1, 31):
    driver.get(url_principal + str(i))
    time.sleep(4)

    categorias_elems = driver.find_elements(By.CSS_SELECTOR, "li.collection--item a") # Cambia por el selector real

    for a in categorias_elems:
        href = a.get_attribute("href")
        if href:
            urls_categorias.append(href)
    
    print(f"Página {i} procesada.")

print("Categorías encontradas:")
for url in urls_categorias:
    print(url)

# 2. Scraping productos de cada categoría
datos = []

for url_cat in urls_categorias:
    driver.get(url_cat)
    time.sleep(5)  # Esperar carga JS

    productos = driver.find_elements(By.CSS_SELECTOR, "div.productitem")
    # en cada categotia crear una carpeta dentro de imagenes_farmacorp para guardar por categoriaa
    categoria_nombre = url_cat.split("/")[-1]
    os.makedirs(f"imagenes_farmacorp/{categoria_nombre}", exist_ok=True)

    for i,p in enumerate(productos):
        try:
            nombre = p.find_element(By.CSS_SELECTOR, "div.productitem--info h2.productitem--title a").text.strip()
            precio = p.find_element(By.CSS_SELECTOR, "div.productitem--info div.price--main span.money").text.strip()
            img_url = p.find_element(By.CSS_SELECTOR, "a.productitem--image-link figure.productitem--image div.image-container img").get_attribute("src")

        # Nombre de archivo para la imagen
            nombre_limpio = nombre.replace(" ", "_").replace("/", "_").replace("\\", "_")[:100]
            ruta_imagen = f"imagenes_farmacorp/{categoria_nombre}/{nombre_limpio}.jpg"

        # Descargar imagen
            img_full_url = "https:" + img_url if img_url.startswith("//") else img_url
            r = requests.get(img_full_url)
            with open(ruta_imagen, "wb") as f:
                f.write(r.content)

            # Guardar datos
            datos.append({
                "categoria": categoria_nombre,
                "nombre": nombre,
                "precio": precio,
                "imagen": ruta_imagen
            })

            # print(f"✅ {i+1}. {nombre} descargado")
            time.sleep(2)  # Espera entre productos

        except Exception as e:
            print(f"❌ Error en producto {i+1}: {e}")
        continue
    
    # cantidad de productos por esta categoría
    print(f"Total de productos procesados del url: {len(datos)}")

driver.quit()

# Crear DataFrame y exportar a CSV
df = pd.DataFrame(datos)
df.to_csv("productos_farmacorp.csv", index=False)
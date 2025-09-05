import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import json

class DigitMapper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.digit_coordinates = {}
        self.current_field = None
        self.current_digit = 0
        
        # Campos con número de dígitos esperados (solo presidentes)
        self.fields_config = {
            'AP': 3,           # 003
            'LYP_ADN': 3,      # 000
            'APB_SUMATE': 3,   # 000
            'LIBRE': 3,        # 000
            'FP': 3,           # 001
            'MAS_IPSP': 3,     # 002
            'MORENA': 3,       # 000
            'UNIDAD': 3,       # 005
            'PDC': 3,          # 008
            'VOTOS_VALIDOS': 3, # 019
            'VOTOS_BLANCOS': 3, # 000
            'VOTOS_NULOS': 3   # 013
        }
        
        self.field_names = list(self.fields_config.keys())
        self.current_field_index = 0
        self.digits_saved = []
        
    def get_current_field_info(self):
        if self.current_field_index >= len(self.field_names):
            return None, None, None
            
        field_name = self.field_names[self.current_field_index]
        total_digits = self.fields_config[field_name]
        
        if field_name not in self.digit_coordinates:
            self.digit_coordinates[field_name] = []
            
        current_digit_num = len(self.digit_coordinates[field_name])
        
        return field_name, current_digit_num, total_digits
        
    def onclick(self, event):
        if event.inaxes is None:
            return
            
        field_info = self.get_current_field_info()
        if field_info[0] is None:
            return
            
        field_name, current_digit_num, total_digits = field_info
        
        if current_digit_num < total_digits:
            x, y = int(event.xdata), int(event.ydata)
            
            # Guardar coordenadas del dígito
            self.digit_coordinates[field_name].append((x, y))
            
            print(f"{field_name} - Dígito {current_digit_num + 1}/{total_digits} -> ({x}, {y})")
            
            # Marcar punto en la imagen
            plt.plot(x, y, 'ro', markersize=6)
            plt.text(x+5, y-10, f'{current_digit_num+1}', fontsize=8, color='blue',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            plt.draw()
            
            # Verificar si completamos el campo actual
            if len(self.digit_coordinates[field_name]) == total_digits:
                print(f"✅ Campo {field_name} completado!")
                self.current_field_index += 1
                
            # Actualizar título
            next_field_info = self.get_current_field_info()
            if next_field_info[0] is not None:
                next_field, next_digit, next_total = next_field_info
                plt.title(f'{next_field} - Dígito {next_digit + 1}/{next_total} (Campo {self.current_field_index + 1}/{len(self.field_names)})')
            else:
                plt.title('¡Mapeo completado! Cierra la ventana.')
                print("\n🎉 ¡Mapeo de todos los dígitos completado!")
                
    def start_digit_mapping(self, figsize=(20, 12)):
        """Inicia el proceso interactivo de mapeo dígito por dígito"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.image_rgb)
        
        # Conectar evento de clic
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        first_field = self.field_names[0]
        first_total = self.fields_config[first_field]
        
        plt.title(f'{first_field} - Dígito 1/{first_total} (Campo 1/{len(self.field_names)})')
        plt.xlabel('Instrucciones: Haz clic en cada DÍGITO individual, de IZQUIERDA a DERECHA')
        plt.tight_layout()
        plt.show()
        
        return self.digit_coordinates
    
    def save_digit_coordinates(self, filename='digit_coordinates.json'):
        """Guarda las coordenadas de cada dígito"""
        with open(filename, 'w') as f:
            json.dump(self.digit_coordinates, f, indent=2)
        print(f"Coordenadas de dígitos guardadas en {filename}")
    
    def extract_individual_digits(self, digit_size=(25, 35), output_dir='extracted_digits'):
        """Extrae cada dígito individual como imagen separada"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        extracted_digits = {}
        digit_count = 0
        
        for field_name, coordinates in self.digit_coordinates.items():
            extracted_digits[field_name] = []
            
            for i, (x, y) in enumerate(coordinates):
                # Definir región del dígito
                x1 = max(0, x - digit_size[0]//2)
                y1 = max(0, y - digit_size[1]//2)
                x2 = min(self.image.shape[1], x + digit_size[0]//2)
                y2 = min(self.image.shape[0], y + digit_size[1]//2)
                
                # Extraer dígito
                digit_img = self.image_rgb[y1:y2, x1:x2]
                
                # Información del dígito
                digit_info = {
                    'image': digit_img,
                    'field': field_name,
                    'position': i,
                    'coordinates': (x, y),
                    'bbox': (x1, y1, x2, y2),
                    'filename': f'{field_name}_digit_{i+1}.png'
                }
                
                extracted_digits[field_name].append(digit_info)
                
                # Guardar imagen individual
                digit_path = os.path.join(output_dir, digit_info['filename'])
                plt.imsave(digit_path, digit_img)
                
                digit_count += 1
                
        print(f"✅ {digit_count} dígitos extraídos en '{output_dir}'")
        return extracted_digits
    
    def visualize_extracted_digits(self, extracted_digits):
        """Visualiza todos los dígitos extraídos organizados por campo"""
        total_digits = sum(len(digits) for digits in extracted_digits.values())
        cols = 10  # dígitos por fila
        rows = (total_digits + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows*2))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        digit_idx = 0
        
        for field_name, digits in extracted_digits.items():
            for i, digit_info in enumerate(digits):
                row = digit_idx // cols
                col = digit_idx % cols
                
                if row < rows and col < cols:
                    axes[row, col].imshow(digit_info['image'])
                    axes[row, col].set_title(f"{field_name}_{i+1}", fontsize=8)
                    axes[row, col].axis('off')
                    
                digit_idx += 1
        
        # Ocultar ejes sobrantes
        for idx in range(digit_idx, rows * cols):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')
                
        plt.tight_layout()
        plt.show()
    
    def create_training_csv(self, extracted_digits, csv_filename='digit_training_data.csv'):
        """Crea CSV con información de cada dígito para entrenamiento"""
        data = []
        
        for field_name, digits in extracted_digits.items():
            for i, digit_info in enumerate(digits):
                data.append({
                    'filename': digit_info['filename'],
                    'field': field_name,
                    'digit_position': i + 1,
                    'x': digit_info['coordinates'][0],
                    'y': digit_info['coordinates'][1],
                    'label': None  # Para llenar manualmente con el valor real
                })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        print(f"📊 CSV creado: {csv_filename}")
        print("📝 Recuerda llenar la columna 'label' con los valores reales de cada dígito")
        return df

# Función principal para usar
def map_digits(image_path):
    """Función para mapear dígitos individuales"""
    mapper = DigitMapper(image_path)
    coordinates = mapper.start_digit_mapping()
    return mapper, coordinates

# Ejemplo de uso completo:
"""
# 1. Mapear dígitos
mapper, coords = map_digits('mesa_1_8004121.jpg')

# 2. Guardar coordenadas
mapper.save_digit_coordinates('digit_coords.json')

# 3. Extraer dígitos individuales
digits = mapper.extract_individual_digits(digit_size=(25, 35))

# 4. Visualizar dígitos extraídos
mapper.visualize_extracted_digits(digits)

# 5. Crear CSV para entrenamiento
df = mapper.create_training_csv(digits, 'training_digits.csv')
"""

print("🎯 Código para mapeo DÍGITO A DÍGITO cargado!")
print("📋 Campos a mapear:", list(DigitMapper('').fields_config.keys()))
print("🚀 Usa: mapper, coords = map_digits('tu_imagen.jpg')")

mapper, coords = map_digits('mesa_1_8004121.jpg')  # Reemplaza con tu imagen

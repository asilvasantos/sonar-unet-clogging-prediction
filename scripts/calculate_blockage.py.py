import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- 1. CONFIGURAÇÕES ---
MODEL_PATH = 'modelo_sonar_unet.h5'

#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_015.png' # Ajuste o nome aqui
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_leve_140.png' # Ajuste o nome aqui
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_moderada_001.png' # Ajuste o nome aqui
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_moderada_037.png'
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_033.png'
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_moderada_050.png' 
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_026.png' 
TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_moderada_038.png' 
#TEST_IMAGE_PATH = 'D:/Dados_Grades_Jirau/dataset/train/sem_anotacao/obstrucao_moderada_038.png' 

PATCH_SIZE = 256

# Cores exatas do seu CVAT (RGB)
COLOR_MAP = np.array([
    [0, 0, 0],         # ID 0: background
    [55, 109, 238],    # ID 1: banco_de_sedimento (Azul)
    [203, 219, 69],    # ID 2: galhos (Verde Limão)
    [60, 223, 245],    # ID 3: obstrucao_leve (Ciano)
    [255, 0, 124],     # ID 4: obstrucao_severa (Rosa/Magenta)
    [36, 179, 83]      # ID 5: sem_obstrucao (Verde Escuro)
], dtype=np.uint8)

# --- 2. CARREGAR MODELO ---
print("Carregando modelo...")
model = load_model(MODEL_PATH)

def predict_full_image(img_path):
     # Carrega imagem original
    img_orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img_orig.shape
    
    # Redimensiona para ser múltiplo do PATCH_SIZE (evita erros no fatiamento)
    new_h = (h // PATCH_SIZE) * PATCH_SIZE
    new_w = (w // PATCH_SIZE) * PATCH_SIZE
    img_resized = cv2.resize(img_orig, (new_w, new_h))
    
    # Prepara a máscara vazia (IDs) e a máscara colorida (RGB)
    full_mask_ids = np.zeros((new_h, new_w), dtype=np.uint8)
    
    print(f"Processando imagem {img_path}...")

    # Fatiamento e Predição por Blocos
    for i in range(0, new_h, PATCH_SIZE):
        for j in range(0, new_w, PATCH_SIZE):
            # Extrai o patch e normaliza
            patch = img_resized[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patch_input = patch.reshape(1, PATCH_SIZE, PATCH_SIZE, 1) / 255.0
            
            # Predição: retorna (1, 256, 256, 6)
            pred = model.predict(patch_input, verbose=0)
            
            # Pega o índice da classe com maior probabilidade (Argmax)
            patch_mask = np.argmax(pred[0], axis=-1)
            
            # Insere o patch na máscara completa
            full_mask_ids[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = patch_mask

    # --- 3. CONVERTER IDS PARA CORES ---
    # Cria uma imagem RGB baseada nos IDs e no COLOR_MAP
    full_mask_rgb = COLOR_MAP[full_mask_ids]
    
    return img_resized, full_mask_rgb

# --- 4. EXIBIÇÃO DOS RESULTADOS ---
img_res, mask_res = predict_full_image(TEST_IMAGE_PATH)

plt.figure(figsize=(16, 8)) # Aumentei um pouco a largura para a legenda caber bem

plt.subplot(1, 2, 1)
plt.title("Imagem Original (Sonar)")
plt.imshow(img_res, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Mapa de Risco (Predição)")
plt.imshow(mask_res)
plt.axis('off')

# --- LEGENDA PERSONALIZADA ATUALIZADA ---
from matplotlib.patches import Patch

legend_elements = [
    # Adicionando a nota sobre o fundo/preto
    Patch(facecolor=COLOR_MAP[0]/255, label='Fundo / Não Identificado'),
    Patch(facecolor=COLOR_MAP[1]/255, label='Banco de Sedimento'),
    Patch(facecolor=COLOR_MAP[2]/255, label='Galhos'),
    Patch(facecolor=COLOR_MAP[3]/255, label='Obstrução Leve'),
    Patch(facecolor=COLOR_MAP[4]/255, label='Obstrução Severa'),
    Patch(facecolor=COLOR_MAP[5]/255, label='Sem Obstrução (Grade Limpa)')
]

# Ajuste da posição da legenda para não cobrir a imagem
plt.legend(handles=legend_elements, 
           loc='center left', 
           bbox_to_anchor=(1.05, 0.5), # Move a legenda para fora da imagem à direita
           title="Classes e Diagnóstico",
           fontsize=10)

plt.tight_layout()

# Salvar o resultado visual antes de fechar
plt.savefig('predicao_final_6classes.png', bbox_inches='tight')
plt.show()
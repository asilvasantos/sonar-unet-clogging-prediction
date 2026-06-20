import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# --- CONFIGURAÇÕES ---
IMG_DIR = 'data/imagens'
MASK_DIR = 'data/masks'
PATCH_SIZE = 256
NUM_CLASSES = 6

COLOR_MAP = [
    [0, 0, 0],         # ID 0: background
    [55, 109, 238],    # ID 1: banco_de_sedimento
    [203, 219, 69],    # ID 2: galhos
    [60, 223, 245],    # ID 3: obstrucao_leve
    [255, 0,  124],    # ID 4: obstrucao_severa
    [36, 179, 83]      # ID 5: sem_obstrucao
]

def map_color_to_id(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_id = np.zeros((h, w), dtype=np.uint8)
    for idx, color in enumerate(COLOR_MAP):
        match = np.all(mask_rgb == color, axis=-1)
        mask_id[match] = idx
    return mask_id

def get_patches(img, mask_id, patch_size):
    patches_img = []
    patches_mask = []
    h, w = img.shape[:2]
    
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch_i = img[i:i+patch_size, j:j+patch_size]
            patch_m = mask_id[i:i+patch_size, j:j+patch_size]
            
            # Normalização da imagem
            patch_i = patch_i / 255.0
            
            patches_img.append(patch_i.reshape(patch_size, patch_size, 1))
            
            # --- CORREÇÃO 1: REMOVIDO O TO_CATEGORICAL ---
            # Para 'sparse_categorical_crossentropy', precisamos apenas do ID (0 a 5)
            # Adicionamos a dimensão de canal (1) no final
            patches_mask.append(patch_m.reshape(patch_size, patch_size, 1))
            
    return patches_img, patches_mask

def load_full_dataset():
    all_x = []
    all_y = []
    filenames = sorted(os.listdir(IMG_DIR))
    print(f"Processando {len(filenames)} imagens grandes...")

    for f in filenames:
        img_path = os.path.join(IMG_DIR, f)
        mask_path = os.path.join(MASK_DIR, f)
        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_rgb = cv2.imread(mask_path)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        p_img, p_mask = get_patches(img, map_color_to_id(mask_rgb), PATCH_SIZE)
        all_x.extend(p_img)
        all_y.extend(p_mask)
        
    return np.array(all_x), np.array(all_y)

X, Y = load_full_dataset()
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)

# --- CORREÇÃO 2: GARANTIR TIPO INTEIRO PARA A MÁSCARA ---
y_train = y_train.astype('int32')
y_val = y_val.astype('int32')

def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    def encoder_block(x, filters):
        s = conv_block(x, filters)
        p = layers.MaxPooling2D((2, 2))(s)
        return s, p

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    b1 = conv_block(p2, 256)

    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    d1 = decoder_block(b1, s2, 128)
    d2 = decoder_block(d1, s1, 64)

    # --- CORREÇÃO 3: VERIFICADO NUM_CLASSES (6) ---
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(d2)
    return models.Model(inputs, outputs, name="UNet_6Classes")

model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 1), num_classes=NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\nIniciando treinamento...")
history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=50, 
                    #batch_size=8
                    batch_size=4)

model.save("models/modelo_sonar_unet.h5")
print("Modelo salvo com sucesso!")
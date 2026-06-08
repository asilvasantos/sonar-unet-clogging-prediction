import os
import sys

# --- CONFIGURAÇÕES DE INICIALIZAÇÃO DO TENSORFLOW (DIRECTML) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURAÇÕES DA PÁGINA WEB ---
st.set_page_config(page_title="SSD - Monitoramento de Grades UHE Jirau", layout="wide")
st.title("⚠️ Sistema de Suporte à Decisão (SSD) - Tomadas de Água")
st.markdown("---")

# --- 1. CARREGAR O MODELO TREINADO (CACHE E ESTABILIDADE) ---
@st.cache_resource
def carregar_modelo_unet():
    MODEL_PATH = "modelo_sonar_unet.h5" 
    try:
        tf.keras.backend.clear_session()
        model = load_model(MODEL_PATH, compile=False) 
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Erro crítico ao carregar o modelo U-Net no DirectML: {e}")
        return None

model = carregar_modelo_unet()

# --- 2. CONFIGURAÇÕES DA BARRA LATERAL (SIDEBAR) ---
st.sidebar.title("🛠️ Painel de Controle")

# Upload do Arquivo
arquivo_upload = st.sidebar.file_uploader("Selecione a imagem do SONAR (ClearVü/SideVü)", type=["png", "jpg", "jpeg"])

# Dicionário mapeando os IDs do seu modelo/CVAT para nomes legíveis
MAPEAMENTO_CLASSES = {
    1: "Banco de Sedimento (ID 1)",
    2: "Galhos / Mat. Lenhoso (ID 2)",
    3: "Obstrução Leve / Moderada (ID 3)",
    4: "Obstrução Severa (ID 4)",
    5: "Sem Obstrução / Grade Limpa (ID 5)"
}

# Configuração da Binarização de Classes para o Cálculo de Tobs
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtro de Binarização (Tobs)")
st.sidebar.info("Selecione quais classes mapeadas pela U-Net devem ser tratadas como 'Obstrução Física' no cálculo percentual:")

classes_selecionadas_nomes = st.sidebar.multiselect(
    "Classes consideradas obstrução:",
    options=list(MAPEAMENTO_CLASSES.values()),
    default=[MAPEAMENTO_CLASSES[1], MAPEAMENTO_CLASSES[2], MAPEAMENTO_CLASSES[4]] # Padrão inicial: Sedimento, Galhos e Severa
)

# Inverte o mapeamento para pegar os IDs numéricos das classes selecionadas
classes_obstrucao = [id_classe for id_classe, nome in MAPEAMENTO_CLASSES.items() if nome in classes_selecionadas_nomes]

# Configurações Hidrológicas do Rio Madeira
st.sidebar.markdown("---")
st.sidebar.subheader("🌊 Parâmetros Hidrológicos")
cenario_hidro = st.sidebar.selectbox(
    "Cenário Sazonal Atual:",
    ("Estiagem / Seca (Vobs = 0.2%/dia)", 
     "Cheia Regular (Vobs = 2.5%/dia)", 
     "Evento Extremo / Repiquete (Vobs = 15.0%/dia)")
)

if "Estiagem" in cenario_hidro:
    vobs = 0.2
elif "Cheia" in cenario_hidro:
    vobs = 2.5
else:
    vobs = 15.0

LIMITE_CRITICO = 80.0

# --- 3. FLUXO PRINCIPAL DA APLICAÇÃO ---
if arquivo_upload is not None:
    imagem_pil_orig = Image.open(arquivo_upload).convert("RGB")
    
    LARGURA_PADRAO = 700
    ALTURA_PADRAO = 500
    imagem_pil = imagem_pil_orig.resize((LARGURA_PADRAO, ALTURA_PADRAO), Image.Resampling.LANCZOS)
    imagem_np = np.array(imagem_pil)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Definição Geométrica da Grade (ROI)")
        st.info("Selecione a ferramenta de retângulo (segundo ícone) no menu do canvas e desenhe o Box sobre a grade.")
        
        canvas_resultado = st_canvas(
            fill_color="rgba(0, 255, 0, 0.15)",  
            stroke_width=2,
            stroke_color="#00FF00",              
            background_image=imagem_pil,
            update_streamlit=True,
            height=ALTURA_PADRAO,
            width=LARGURA_PADRAO,
            drawing_mode="rect",
            key="canvas_sonar",
        )

    if canvas_resultado.json_data is not None and len(canvas_resultado.json_data["objects"]) > 0:
        with col2:
            st.subheader("2. Análise e Diagnóstico Ciberfísico")
            
            if st.button("Executar Segmentação e Calcular Indicadores"):
                if len(classes_obstrucao) == 0:
                    st.error("⚠️ Erro: Selecione pelo menos uma classe na barra lateral para servir de base para o cálculo de obstrução.")
                else:
                    with st.spinner("U-Net processando matriz acústica..."):
                        
                        # --- PRÉ-PROCESSAMENTO ---
                        img_cinza = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2GRAY)
                        img_input = cv2.resize(img_cinza, (256, 256))
                        img_input = np.expand_dims(img_input, axis=-1)  
                        img_input = np.expand_dims(img_input, axis=0)   
                        img_input = img_input / 255.0                   
                        
                        # --- INFERÊNCIA ---
                        predicao = model.predict(img_input)
                        mapa_segmentado = np.argmax(predicao, axis=-1)[0] 
                        
                        mapa_original_res = cv2.resize(mapa_segmentado.astype('uint8'), 
                                                       (LARGURA_PADRAO, ALTURA_PADRAO), 
                                                       interpolation=cv2.INTER_NEAREST)
                        
                        # --- MÁSCARA DA ROI ---
                        mascara_roi = np.zeros((ALTURA_PADRAO, LARGURA_PADRAO), dtype=np.uint8)
                        for obj in canvas_resultado.json_data["objects"]:
                            if obj["type"] == "rect":
                                x = int(obj["left"])
                                y = int(obj["top"])
                                w = int(obj["width"])
                                h = int(obj["height"])
                                cv2.rectangle(mascara_roi, (x, y), (x + w, y + h), 1, -1)
                        
                        # --- CÁLCULO DINÂMICO DO TOBS COM AS CLASSES ESCOLHIDAS ---
                        pixels_totais_grade = np.sum(mascara_roi == 1)
                        
                        pixels_obstruidos_roi = np.zeros_like(mapa_original_res)
                        for classe in classes_obstrucao:
                            pixels_obstruidos_roi[(mapa_original_res == classe) & (mascara_roi == 1)] = 1
                        
                        total_pixels_obstruidos = np.sum(pixels_obstruidos_roi == 1)
                        
                        if pixels_totais_grade > 0:
                            tobs = (total_pixels_obstruidos / pixels_totais_grade) * 100
                        else:
                            tobs = 0.0
                        
                        # --- CÁLCULO PREDITIVO (TMAN) ---
                        if tobs >= LIMITE_CRITICO:
                            tman = 0.0
                        else:
                            tman = (LIMITE_CRITICO - tobs) / vobs
                        
                        # --- EXIBIÇÃO DASHBOARD ---
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric(label="Taxa de Obstrução (Tobs)", value=f"{tobs:.2f} %")
                        with col_m2:
                            st.metric(label="Fator de Aporte (Vobs)", value=f"{vobs:.2f} % / dia")
                        with col_m3:
                            if tman == 0:
                                st.metric(label="Tempo para Manutenção (Tman)", value="IMEDIATO", delta="-Crítico")
                            elif tman < 1.0:
                                st.metric(label="Tempo para Manutenção (Tman)", value=f"{tman*24:.1f} Horas")
                            else:
                                st.metric(label="Tempo para Manutenção (Tman)", value=f"{tman:.1f} Dias")
                        
                        # --- DIAGNÓSTICO DO SSD ---
                        st.markdown("### 📋 Diagnóstico Operacional e Diretrizes")
                        st.caption(f"Cálculo baseado nas classes binarizadas: {', '.join([MAPEAMENTO_CLASSES[c] for c in classes_obstrucao])}")
                        
                        if tobs >= LIMITE_CRITICO:
                            st.error(f"🚨 **CRÍTICO:** A grade superou o limite de {LIMITE_CRITICO}%. Risco hidráulico severo. Parada preventiva recomendada.")
                        elif tman <= 3.0:
                            st.error(f"🔴 **ALERTA SEVERO:** Janela estreita. Colmatação máxima estimada em {tman:.1f} dias. Mobilizar limpeza física.")
                        elif tman <= 7.0:
                            st.warning(f"🟡 **ATENÇÃO:** Margem operacional decrescendo ({tman:.1f} dias restantes). Planejar intervenção para esta semana.")
                        else:
                            st.success(f"🟢 **OPERAÇÃO SEGURA:** Nível sob controle. Próxima intervenção estimada em {tman:.1f} dias.")
                        
                        # --- OVERLAY COLORIDO DO MAPA DE RISCO ---
                        COLOR_MAP = np.array([
                            [0, 0, 0],         # ID 0: background
                            [55, 109, 238],    # ID 1: banco_de_sedimento (Azul)
                            [203, 219, 69],    # ID 2: galhos (Verde Limão)
                            [60, 223, 245],    # ID 3: obstrucao_leve (Ciano)
                            [255, 0, 124],     # ID 4: obstrucao_severa (Rosa/Magenta)
                            [36, 179, 83]      # ID 5: sem_obstrucao (Verde Escuro)
                        ], dtype=np.uint8)
                        
                        imagem_segmentada_colorida = COLOR_MAP[mapa_original_res]
                        imagem_segmentada_colorida[mascara_roi == 0] = [0, 0, 0]
                        
                        imagem_overlay = cv2.addWeighted(imagem_np, 0.7, imagem_segmentada_colorida, 0.3, 0)
                        st.image(imagem_overlay, caption="Mapa de Risco Gerado pela U-Net (Restrito à ROI)", use_column_width=True)
    else:
        with col2:
            st.info("Aguardando a definição do Box delimitador da grade na imagem da esquerda.")
else:
    st.info("Por favor, faça o upload de uma imagem acústica no menu lateral para iniciar o monitoramento.")
import streamlit as st
import cv2
import tempfile
import math
import numpy as np
import os
from ultralytics import YOLO
from sort import *

# ==========================================
# 1. CONFIGURAÇÃO VISUAL E CSS
# ==========================================
st.set_page_config(page_title="VigiIA - Monitoramento", layout="wide", page_icon="👁️")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 350px;}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 350px; margin-left: -350px}
    
    /* Estilo dos Cards de KPI */
    .kpi-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 16px;
        font-weight: bold;
        color: #555;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# ==========================================
def process_video(video_path, confidence_threshold, classes_to_track, line_height_perc, sensitivity):
    
    model = YOLO('yolov8n.pt') 
    classNames = model.names
    
    tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Erro ao abrir o vídeo: {video_path}")
        return

    # ---------------- Layout dos KPIs ----------------
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        kpi1_placeholder = st.empty() 
        kpi1_placeholder.markdown('<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">0</div></div>', unsafe_allow_html=True)
        
    with kpi2:
        kpi2_placeholder = st.empty()
        kpi2_placeholder.markdown('<div class="kpi-card"><div class="kpi-title">Resolução</div><div class="kpi-value">...</div></div>', unsafe_allow_html=True)

    with kpi3:
        kpi3_placeholder = st.empty() 
        kpi3_placeholder.markdown('<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value">0</div></div>', unsafe_allow_html=True)
        
    stframe = st.empty()
    totalCount = []

    ret, frame = cap.read()
    if not ret:
        st.error("Não foi possível ler o primeiro frame do vídeo.")
        return

    height, width, _ = frame.shape
    kpi2_placeholder.markdown(f'<div class="kpi-card"><div class="kpi-title">Resolução</div><div class="kpi-value">{width}x{height}</div></div>', unsafe_allow_html=True)

    line_pos = int(height * (line_height_perc / 100))
    offset = sensitivity 

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True, verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in classes_to_track and conf > confidence_threshold:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)
        
        kpi1_placeholder.markdown(f'<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">{len(resultsTracker)}</div></div>', unsafe_allow_html=True)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            
            cx, cy = x1 + w // 2, y1 + h // 2
            color_box = (255, 0, 255) 
            
            if line_pos - offset < cy < line_pos + offset:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    color_box = (0, 255, 0)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(img, f'ID: {int(id)}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        kpi3_placeholder.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value" style="color:red">{len(totalCount)}</div></div>', unsafe_allow_html=True)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB", use_container_width=True)

    cap.release()

# ==========================================
# 3. INTERFACE LATERAL
# ==========================================
def main():
    st.title("VigiIA - Sistema de Contagem Inteligente")
    st.markdown("SECTI - Secretaria Municipal de Ciência, Tecnologia e Inovação")
    st.markdown("---")

    st.sidebar.header("⚙️ Configurações")
    
    default_classes = ["person", "car", "bus", "truck", "motorbike"]
    assigned_class = st.sidebar.multiselect("O que você quer contar?", default_classes, default=["person"])
    
    confidence = st.sidebar.slider('Confiança da IA', 0.0, 1.0, 0.40)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Calibragem da Linha")
    
    # AQUI ESTÁ O TEXTO AZUL QUE VOCÊ PEDIU DE VOLTA
    st.sidebar.info("Ajuste a posição até a contagem funcionar da melhor forma.")
    
    line_height = st.sidebar.slider('Posição da Linha (Vertical)', 0, 100, 50)
    sensitivity = st.sidebar.slider('Sensibilidade da Linha', 10, 100, 40)

    st.sidebar.markdown("---")
    
    DEFAULT_VIDEO = "video_padrao.mp4"
    use_default = False
    video_path = ""

    video_file_buffer = st.sidebar.file_uploader("Carregar Vídeo (MP4, AVI)", type=["mp4", "mov", "avi"])
    
    if video_file_buffer:
        tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tfflie.write(video_file_buffer.read())
        video_path = tfflie.name
        st.sidebar.success("Vídeo do usuário carregado!")
    
    else:
        if os.path.exists(DEFAULT_VIDEO):
            video_path = DEFAULT_VIDEO
            use_default = True
            st.sidebar.info(f"Usando vídeo padrão: {DEFAULT_VIDEO}")
        else:
            st.sidebar.warning(f"Nenhum vídeo carregado e '{DEFAULT_VIDEO}' não encontrado.")

    if video_path:
        if st.sidebar.button("▶️ Iniciar Contagem", type="primary"):
            process_video(video_path, confidence, assigned_class, line_height, sensitivity)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
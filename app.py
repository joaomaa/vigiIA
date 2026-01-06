import streamlit as st
import cv2
import tempfile
import math
import numpy as np
import os
import av
import time
from ultralytics import YOLO
from sort import *
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ==========================================
# 1. CONFIGURAÇÃO VISUAL E CSS
# ==========================================
st.set_page_config(page_title="VigiIA - Monitoramento", layout="wide", page_icon="👁️")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 350px;}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 350px; margin-left: -350px}
    
    /* Estilo dos Cards (KPIs) */
    .kpi-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-title { font-size: 16px; font-weight: bold; color: #555; }
    .kpi-value { font-size: 32px; font-weight: bold; color: #000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicializar estado da webcam
if "webcam_ativa" not in st.session_state:
    st.session_state["webcam_ativa"] = False

# ==========================================
# 2. CLASSE PROCESSADORA (WEBCAM)
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
        self.totalCount = []
        
        # Variáveis compartilhadas
        self.active_count = 0
        self.frame_width = 0
        self.frame_height = 0
        
        self.confidence_threshold = 0.4
        self.classes_to_track = ["person", "car", "bus", "truck", "motorbike"]
        self.line_height_perc = 50
        self.sensitivity = 40
        self.model_names = self.model.names

    def update_settings(self, conf, classes, line, sens):
        self.confidence_threshold = conf
        self.classes_to_track = classes
        self.line_height_perc = line
        self.sensitivity = sens

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        height, width, _ = img.shape
        self.frame_width = width
        self.frame_height = height
        
        line_pos = int(height * (self.line_height_perc / 100))
        offset = self.sensitivity

        # --- DETECÇÃO ---
        results = self.model(img, stream=True, verbose=False)
        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = self.model_names[cls]

                if currentClass in self.classes_to_track and conf > self.confidence_threshold:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # --- RASTREAMENTO ---
        resultsTracker = self.tracker.update(detections)
        self.active_count = len(resultsTracker)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            color_box = (255, 0, 255) # Magenta

            # Lógica de Contagem (Invisível)
            if line_pos - offset < cy < line_pos + offset:
                if self.totalCount.count(id) == 0:
                    self.totalCount.append(id)
                    color_box = (0, 255, 0)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(img, f'{int(id)}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. FUNÇÃO ARQUIVO DE VÍDEO
# ==========================================
def process_video_file(video_path, confidence_threshold, classes_to_track, line_height_perc, sensitivity):
    model = YOLO('yolov8n.pt') 
    classNames = model.names
    tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"Erro ao abrir vídeo: {video_path}")
        return

    # Layout KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1: kpi1_ph = st.empty()
    with kpi2: kpi2_ph = st.empty()
    with kpi3: kpi3_ph = st.empty()
    
    stframe = st.empty()
    totalCount = []

    ret, frame = cap.read()
    if not ret: return
    height, width, _ = frame.shape
    kpi2_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Resolução</div><div class="kpi-value">{width}x{height}</div></div>', unsafe_allow_html=True)
    
    line_pos = int(height * (line_height_perc / 100))
    offset = sensitivity 

    # Botão de Parar (Aparece abaixo do iniciar enquanto roda)
    stop_bt = st.sidebar.button("⏹️ Parar Vídeo")

    while cap.isOpened() and not stop_bt:
        success, img = cap.read()
        if not success: break
        
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
        kpi1_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">{len(resultsTracker)}</div></div>', unsafe_allow_html=True)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            color_box = (255, 0, 255)
            
            # Lógica VISÍVEL para arquivo de vídeo
            if line_pos - offset < cy < line_pos + offset:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    color_box = (0, 255, 0)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    cv2.line(img, (0, line_pos), (width, line_pos), (0, 255, 0), 4)

            if color_box != (0, 255, 0):
                 cv2.line(img, (0, line_pos), (width, line_pos), (0, 0, 255), 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(img, f'{int(id)}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
        
        if len(resultsTracker) == 0:
             cv2.line(img, (0, line_pos), (width, line_pos), (0, 0, 255), 2)

        kpi3_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value" style="color:red">{len(totalCount)}</div></div>', unsafe_allow_html=True)
        stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()

# ==========================================
# 4. INTERFACE PRINCIPAL
# ==========================================
def main():
    st.title("VigiIA - Sistema de Contagem Inteligente")
    st.markdown("Secretaria Municipal de Ciência, Tecnologia e Inovação")
    st.markdown("---")
    
    DEFAULT_VIDEO = "video_padrao.mp4"
    final_video_path = None

    # --- MENU 1: FONTE ---
    with st.sidebar.expander("📹 Fonte de Vídeo", expanded=True):
        source_radio = st.radio("Escolha a Entrada:", ["Arquivo de Vídeo", "Webcam (Ao Vivo)"])
        
        video_file_buffer = None
        if source_radio == "Arquivo de Vídeo":
            st.caption("Faça upload ou use o vídeo demonstrativo automático.")
            video_file_buffer = st.file_uploader("Upload MP4/AVI (Opcional)", type=["mp4", "avi", "mov"])

    st.sidebar.markdown("---")

    # --- MENU 2: CONFIGURAÇÕES IA ---
    st.sidebar.header("🧠 Inteligência Artificial")
    default_classes = ["person", "car", "bus", "truck", "motorbike"]
    assigned_class = st.sidebar.multiselect("Objetos para contar:", default_classes, default=["person"])
    confidence = st.sidebar.slider('Confiança Mínima', 0.0, 1.0, 0.40)
    
    st.sidebar.markdown("---")
    
    # --- MENU 3: CALIBRAGEM ---
    st.sidebar.header("📏 Calibragem da Linha")
    st.sidebar.info("Ajuste a linha azul virtual para onde o fluxo de pessoas/veículos passa.")
    line_height = st.sidebar.slider('Posição Vertical (%)', 0, 100, 50)
    sensitivity = st.sidebar.slider('Sensibilidade de Detecção', 10, 100, 40)

    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Execução")

    # ===============================
    # 1. MODO WEBCAM
    # ===============================
    if source_radio == "Webcam (Ao Vivo)":
        
        # --- LÓGICA DOS BOTÕES IGUAL A IMAGEM ---
        
        # 1. Botão Iniciar (Vermelho/Primary)
        # Sempre visível para reiniciar ou começar
        if st.sidebar.button("▶️ Iniciar Processamento", type="primary"):
            st.session_state["webcam_ativa"] = True
            st.rerun() # Atualiza a tela para processar
            
        # 2. Botão Parar (Branco/Secondary)
        # Só aparece (empilhado embaixo) se a webcam estiver ativa
        if st.session_state["webcam_ativa"]:
            if st.sidebar.button("⏹️ Parar Vídeo"):
                st.session_state["webcam_ativa"] = False
                st.rerun()

        # Layout dos Cards
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1: kpi1_ph = st.empty()
        with kpi2: kpi2_ph = st.empty()
        with kpi3: kpi3_ph = st.empty()

        # Placeholders iniciais
        kpi1_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">0</div></div>', unsafe_allow_html=True)
        kpi2_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Resolução</div><div class="kpi-value">-</div></div>', unsafe_allow_html=True)
        kpi3_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value" style="color:red">0</div></div>', unsafe_allow_html=True)

        if st.session_state["webcam_ativa"]:
            st.header("Webcam em Tempo Real")
            
            ctx = webrtc_streamer(
                key="vigia-webcam", 
                mode=WebRtcMode.SENDRECV, 
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                desired_playing_state=st.session_state["webcam_ativa"]
            )
            
            # Loop de atualização dos KPIs
            if ctx.video_processor:
                ctx.video_processor.update_settings(confidence, assigned_class, line_height, sensitivity)
                
                while ctx.state.playing:
                    ativos = ctx.video_processor.active_count
                    largura = ctx.video_processor.frame_width
                    altura = ctx.video_processor.frame_height
                    total = len(ctx.video_processor.totalCount)
                    
                    kpi1_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">{ativos}</div></div>', unsafe_allow_html=True)
                    if largura > 0:
                        kpi2_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Resolução</div><div class="kpi-value">{largura}x{altura}</div></div>', unsafe_allow_html=True)
                    kpi3_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value" style="color:red">{total}</div></div>', unsafe_allow_html=True)
                    
                    time.sleep(0.5)
        else:
            st.info("Clique em 'Iniciar Processamento' na barra lateral para ativar a câmera.")

    # ===============================
    # 2. MODO ARQUIVO DE VÍDEO
    # ===============================
    elif source_radio == "Arquivo de Vídeo":
        
        if video_file_buffer:
            tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tfflie.write(video_file_buffer.read())
            final_video_path = tfflie.name
            st.success("Vídeo carregado com sucesso!")
        else:
            if os.path.exists(DEFAULT_VIDEO):
                final_video_path = DEFAULT_VIDEO
                st.info(f"Nenhum arquivo enviado. Utilizando vídeo de demonstração padrão.")
            else:
                st.warning("Aguardando upload... (Vídeo padrão 'video_padrao.mp4' não encontrado)")
        
        btn_disabled = final_video_path is None

        # Botão Iniciar (Mesmo estilo)
        if st.sidebar.button("▶️ Iniciar Processamento", type="primary", disabled=btn_disabled):
            if final_video_path:
                process_video_file(final_video_path, confidence, assigned_class, line_height, sensitivity)
            else:
                st.sidebar.error("Nenhum arquivo de vídeo disponível.")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
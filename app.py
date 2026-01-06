import streamlit as st
import cv2
import tempfile
import math
import numpy as np
import os
import av
import time
import pandas as pd
from datetime import datetime
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

# Inicializar estado da execução
if "run_camera" not in st.session_state:
    st.session_state["run_camera"] = False

# Arquivo para salvar histórico
CSV_FILE = "historico_contagem.csv"

# ==========================================
# 2. FUNÇÃO DE PERSISTÊNCIA (CSV)
# ==========================================
def salvar_contagem(classe_obj):
    """Salva a data, hora e classe detectada no CSV."""
    agora = datetime.now()
    novo_dado = pd.DataFrame({
        "Data": [agora.strftime("%Y-%m-%d")],
        "Hora": [agora.strftime("%H:%M:%S")],
        "Classe": [classe_obj],
        "Contagem": [1]
    })
    
    if not os.path.isfile(CSV_FILE):
        novo_dado.to_csv(CSV_FILE, index=False)
    else:
        novo_dado.to_csv(CSV_FILE, mode='a', header=False, index=False)

# ==========================================
# 3. CLASSE PROCESSADORA (WEBCAM)
# ==========================================
class VideoProcessor:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
        self.totalCount = []
        
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

        resultsTracker = self.tracker.update(detections)
        self.active_count = len(resultsTracker)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            color_box = (255, 0, 255)

            # Lógica Invisível
            if line_pos - offset < cy < line_pos + offset:
                if self.totalCount.count(id) == 0:
                    self.totalCount.append(id)
                    salvar_contagem("Webcam_Detect") # Salva no CSV
                    color_box = (0, 255, 0)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(img, f'{int(id)}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. FUNÇÃO GENÉRICA (VÍDEO E IP CAMERA)
# ==========================================
def process_stream_source(source_path, confidence_threshold, classes_to_track, line_height_perc, sensitivity, is_ip_camera=False):
    """Processa tanto arquivos de vídeo quanto Streams RTSP/HTTP"""
    model = YOLO('yolov8n.pt') 
    classNames = model.names
    tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
    
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        st.error(f"Erro ao conectar na fonte: {source_path}")
        st.session_state["run_camera"] = False
        return

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

    # Botão de Parar (Controlado pelo Session State ou Botão Sidebar)
    # Para Video/IP, verificamos o session_state definido pelos botões
    
    while cap.isOpened() and st.session_state["run_camera"]:
        success, img = cap.read()
        if not success:
            if not is_ip_camera: # Se for arquivo e acabou
                break
            else: # Se for IP e caiu, tenta reconectar ou sai
                st.warning("Sinal da câmera perdido.")
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
        kpi1_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Ativos Agora</div><div class="kpi-value">{len(resultsTracker)}</div></div>', unsafe_allow_html=True)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            color_box = (255, 0, 255)
            
            if line_pos - offset < cy < line_pos + offset:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    salvar_contagem(currentClass if 'currentClass' in locals() else "Objeto")
                    color_box = (0, 255, 0)
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    cv2.line(img, (0, line_pos), (width, line_pos), (0, 255, 0), 4)

            # Desenha linha VISÍVEL (Padrão para Arquivo e IP Camera)
            if color_box != (0, 255, 0):
                 cv2.line(img, (0, line_pos), (width, line_pos), (0, 0, 255), 2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(img, f'{int(id)}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
        
        if len(resultsTracker) == 0:
             cv2.line(img, (0, line_pos), (width, line_pos), (0, 0, 255), 2)

        kpi3_ph.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Contado</div><div class="kpi-value" style="color:red">{len(totalCount)}</div></div>', unsafe_allow_html=True)
        stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    st.session_state["run_camera"] = False # Reseta estado ao sair

# ==========================================
# 5. DASHBOARD (GRÁFICOS)
# ==========================================
def show_dashboard():
    st.header("📊 Histórico de Detecções")
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        
        # Filtros
        col1, col2 = st.columns(2)
        with col1:
            datas = df['Data'].unique()
            data_selecionada = st.selectbox("Selecione a Data:", datas)
        
        df_filtered = df[df['Data'] == data_selecionada]
        
        # Gráfico de Barras por Hora
        st.subheader(f"Fluxo Horário - {data_selecionada}")
        df_filtered['Hora_Short'] = df_filtered['Hora'].apply(lambda x: x.split(':')[0] + 'h')
        hourly_counts = df_filtered.groupby('Hora_Short')['Contagem'].sum().reset_index()
        st.bar_chart(hourly_counts, x="Hora_Short", y="Contagem", color="#FF4B4B")
        
        # Métricas Totais
        total_dia = df_filtered['Contagem'].sum()
        st.metric("Total Detectado no Dia", total_dia)
        
        with st.expander("Ver Dados Brutos"):
            st.dataframe(df_filtered)
    else:
        st.info("Nenhum dado histórico encontrado ainda. Inicie o monitoramento para gerar dados.")

# ==========================================
# 6. INTERFACE PRINCIPAL
# ==========================================
def main():
    st.title("VigiIA - Sistema de Contagem Inteligente")
    st.markdown("Secretaria Municipal de Ciência, Tecnologia e Inovação")
    
    # Abas principais
    tab_monitor, tab_dash = st.tabs(["🎥 Monitoramento", "📈 Relatórios"])

    with tab_monitor:
        st.markdown("---")
        DEFAULT_VIDEO = "video_padrao.mp4"
        final_source = None
        is_ip_cam = False

        # --- MENU 1: FONTE ---
        with st.sidebar.expander("📹 Fonte de Vídeo", expanded=True):
            source_radio = st.radio("Escolha a Entrada:", 
                                  ["Arquivo de Vídeo", "Webcam (Ao Vivo)", "Câmera IP (RTSP)"])
            
            video_file_buffer = None
            rtsp_url = None

            if source_radio == "Arquivo de Vídeo":
                st.caption("Faça upload ou use o vídeo demonstrativo.")
                video_file_buffer = st.file_uploader("Upload MP4/AVI", type=["mp4", "avi"])
            
            elif source_radio == "Câmera IP (RTSP)":
                st.caption("Ex: rtsp://admin:12345@192.168.0.1:554/stream")
                rtsp_url = st.text_input("URL da Câmera:")

        st.sidebar.markdown("---")

        # --- MENU 2: CONFIGURAÇÕES IA ---
        st.sidebar.header("🧠 Inteligência Artificial")
        default_classes = ["person", "car", "bus", "truck", "motorbike"]
        assigned_class = st.sidebar.multiselect("Objetos para contar:", default_classes, default=["person"])
        confidence = st.sidebar.slider('Confiança Mínima', 0.0, 1.0, 0.40)
        
        st.sidebar.markdown("---")
        
        # --- MENU 3: CALIBRAGEM ---
        st.sidebar.header("📏 Calibragem da Linha")
        line_height = st.sidebar.slider('Posição Vertical (%)', 0, 100, 50)
        sensitivity = st.sidebar.slider('Sensibilidade', 10, 100, 40)

        st.sidebar.markdown("---")
        st.sidebar.header("🚀 Execução")

        # =====================================
        # LÓGICA DE CONTROLE (BOTÕES SIDEBAR)
        # =====================================
        
        # Botões Iniciar/Parar Padrão para TODOS os modos (Exceto WebRTC que tem controle próprio interno)
        # Vamos padronizar: Botões controlam o st.session_state["run_camera"]
        
        col_btn1, col_btn2 = st.sidebar.columns(2)
        
        # Só mostramos os botões se NÃO for WebRTC (ele tem lógica separada para usar a API nativa)
        # Mas para ficar uniforme visualmente, faremos um wrapper.
        
        btn_iniciar = False
        btn_parar = False
        
        # Botão Iniciar (Vermelho)
        if st.sidebar.button("▶️ Iniciar Processamento", type="primary"):
            st.session_state["run_camera"] = True
            st.rerun()

        # Botão Parar (Branco) - Só aparece se estiver rodando
        if st.session_state["run_camera"]:
            if st.sidebar.button("⏹️ Parar Vídeo"):
                st.session_state["run_camera"] = False
                st.rerun()

        # ===============================
        # PROCESSAMENTO
        # ===============================
        
        if st.session_state["run_camera"]:
            
            # 1. ARQUIVO DE VÍDEO
            if source_radio == "Arquivo de Vídeo":
                if video_file_buffer:
                    tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                    tfflie.write(video_file_buffer.read())
                    final_source = tfflie.name
                elif os.path.exists(DEFAULT_VIDEO):
                    final_source = DEFAULT_VIDEO
                
                if final_source:
                    process_stream_source(final_source, confidence, assigned_class, line_height, sensitivity, is_ip_camera=False)
                else:
                    st.error("Nenhum vídeo carregado.")
                    st.session_state["run_camera"] = False

            # 2. CÂMERA IP
            elif source_radio == "Câmera IP (RTSP)":
                if rtsp_url:
                    # Para câmeras de segurança, mantemos a linha VISÍVEL (igual arquivo)
                    process_stream_source(rtsp_url, confidence, assigned_class, line_height, sensitivity, is_ip_camera=True)
                else:
                    st.error("Insira a URL RTSP da câmera.")
                    st.session_state["run_camera"] = False

            # 3. WEBCAM (Lógica Especial WebRTC)
            elif source_radio == "Webcam (Ao Vivo)":
                st.header("Webcam em Tempo Real")
                
                # KPIs Placeholder
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1: kpi1_ph = st.empty()
                with kpi2: kpi2_ph = st.empty()
                with kpi3: kpi3_ph = st.empty()
                
                ctx = webrtc_streamer(
                    key="vigia-webcam", 
                    mode=WebRtcMode.SENDRECV, 
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    desired_playing_state=st.session_state["run_camera"] # Controlado pelo botão da sidebar
                )
                
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
            st.info("Aguardando início do processamento...")

    # Aba de Dashboard
    with tab_dash:
        show_dashboard()

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
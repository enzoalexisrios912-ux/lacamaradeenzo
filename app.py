import math
import time
import cv2
import numpy as np
import threading
import urllib.request
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Sistema Profesional de Monitoreo de Ganado y Clima")

# Cargar modelo YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Estado compartido en memoria con métricas ampliadas
system_state = {
    "cow_count": 0,          
    "total_ingresos": 0,     
    "total_egresos": 0,      
    "vacas_adentro": 0,      
    "fps": 0.0,              
    "status": "Iniciando...",
    "temperature": "--",     
    "humidity": "--",        
    "weather_time": "--",
    "uptime_seconds": 0
}
lock = threading.Lock()
start_time = time.time()

# Configuración de cámara y video
CAMERA_SOURCE = 1
FLIP_CAMERA_HORIZONTAL = True
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
YOLO_IMGSZ = 480
JPEG_QUALITY = 90

# Configuración de la cerca
LINE_P1_FRAC = (0.0, 0.6)
LINE_P2_FRAC = (1.0, 0.6)
INSIDE_SIGN = 1
CROSS_MARGIN_PX = 15
CONFIRM_FRAMES = 3

FENCE_IMAGE_PATH = "fence.png"   
FENCE_HEIGHT_PX = 70             

track_side = {}      
pending_side = {}     
pending_count = {}    
inside_ids = set()    


def fetch_weather_loop():
    """Consulta la API de Open-Meteo para Posadas, Misiones cada 5 minutos."""
    url = "https://api.open-meteo.com/v1/forecast?latitude=-27.3671&longitude=-55.8961&current=temperature_2m,relative_humidity_2m&timezone=America/Argentina/Buenos_Aires"
    while True:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                current = data.get("current", {})
                temp = current.get("temperature_2m")
                hum = current.get("relative_humidity_2m")
                time_str = current.get("time")
                
                with lock:
                    system_state["temperature"] = f"{temp}" if temp is not None else "--"
                    system_state["humidity"] = f"{hum}" if hum is not None else "--"
                    system_state["weather_time"] = time_str.replace("T", " ") if time_str else "--"
        except Exception as e:
            print(f"Error al actualizar el clima: {e}")
        
        time.sleep(300)


@app.on_event("startup")
def startup_event():
    weather_thread = threading.Thread(target=fetch_weather_loop, daemon=True)
    weather_thread.start()


def signed_distance_to_line(px, py, x1, y1, x2, y2):
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return cross / length


def load_fence_tile():
    tile = cv2.imread(FENCE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if tile is None:
        return None
    if tile.shape[2] == 3:  
        alpha = np.full(tile.shape[:2], 255, dtype=tile.dtype)
        tile = np.dstack([tile, alpha])
    tile = cv2.flip(tile, 0)
    return tile


def build_fence_strip(tile, length_px, height_px):
    th, tw = tile.shape[:2]
    scale = height_px / th
    new_w = max(1, int(round(tw * scale)))
    tile_resized = cv2.resize(tile, (new_w, height_px), interpolation=cv2.INTER_AREA)
    reps = int(math.ceil(length_px / new_w)) + 1
    tiled = np.tile(tile_resized, (1, reps, 1))
    return tiled[:, :max(1, int(length_px)), :]


def rotate_rgba(img, angle_deg):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def overlay_rgba(background, overlay, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1e, y1e = min(x + ow, bw), min(y + oh, bh)
    if x0 >= x1e or y0 >= y1e:
        return
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1e - x0), oy0 + (y1e - y0)
    overlay_crop = overlay[oy0:oy1, ox0:ox1]
    alpha = (overlay_crop[:, :, 3:4].astype(np.float32)) / 255.0
    bg_region = background[y0:y1e, x0:x1e].astype(np.float32)
    blended = overlay_crop[:, :, :3].astype(np.float32) * alpha + bg_region * (1 - alpha)
    background[y0:y1e, x0:x1e] = blended.astype(np.uint8)


class CameraStream:
    def __init__(self, source, width, height, fps):
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None

    def is_opened(self):
        return self.cap.isOpened()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


def get_video_stream():
    camera = CameraStream(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    if not camera.is_opened():
        with lock:
            system_state["status"] = "Error al abrir la cámara"
        return
    camera.start()

    with lock:
        system_state["status"] = "Monitoreando en vivo"

    line_pixels = None       
    fence_overlay = None     
    fence_pos = (0, 0)       
    fence_tile = load_fence_tile()
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    fps_smoothed = 0.0
    last_tick = time.time()
    frames_without_camera = 0

    while True:
        frame = camera.read()
        if frame is None:
            frames_without_camera += 1
            if frames_without_camera > 150: 
                with lock:
                    system_state["status"] = "Señal perdida"
                break
            time.sleep(0.01)
            continue
        frames_without_camera = 0

        if FLIP_CAMERA_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        if line_pixels is None:
            x1 = int(LINE_P1_FRAC[0] * w)
            y1 = int(LINE_P1_FRAC[1] * h)
            x2 = int(LINE_P2_FRAC[0] * w)
            y2 = int(LINE_P2_FRAC[1] * h)
            line_pixels = (x1, y1, x2, y2)

            if fence_tile is not None:
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                strip = build_fence_strip(fence_tile, length, FENCE_HEIGHT_PX)
                fence_overlay = rotate_rgba(strip, angle)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                fence_pos = (
                    int(mx - fence_overlay.shape[1] / 2),
                    int(my - fence_overlay.shape[0] / 2),
                )

        lx1, ly1, lx2, ly2 = line_pixels

        results = model.track(frame, classes=[19], conf=0.45, persist=True, verbose=False, imgsz=YOLO_IMGSZ)
        boxes = results[0].boxes
        current_cows = len(boxes)

        seen_ids = set()
        if boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()
            seen_ids = set(ids)

            for track_id, (bx1, by1, bx2, by2) in zip(ids, xyxy):
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                dist = signed_distance_to_line(cx, cy, lx1, ly1, lx2, ly2) * INSIDE_SIGN

                if dist > CROSS_MARGIN_PX:
                    observed_side = True   
                elif dist < -CROSS_MARGIN_PX:
                    observed_side = False  
                else:
                    observed_side = None   

                if observed_side is None:
                    continue

                confirmed_side = track_side.get(track_id)
                if confirmed_side is None:
                    track_side[track_id] = observed_side
                    pending_side[track_id] = None
                    pending_count[track_id] = 0
                    if observed_side:
                        inside_ids.add(track_id)
                    continue

                if observed_side == confirmed_side:
                    pending_side[track_id] = None
                    pending_count[track_id] = 0
                    continue

                if pending_side.get(track_id) == observed_side:
                    pending_count[track_id] = pending_count.get(track_id, 0) + 1
                else:
                    pending_side[track_id] = observed_side
                    pending_count[track_id] = 1

                if pending_count[track_id] < CONFIRM_FRAMES:
                    continue

                track_side[track_id] = observed_side
                pending_side[track_id] = None
                pending_count[track_id] = 0

                if observed_side:
                    inside_ids.add(track_id)
                    with lock:
                        system_state["total_ingresos"] += 1
                else:
                    inside_ids.discard(track_id)
                    with lock:
                        system_state["total_egresos"] += 1

        for stale_id in list(pending_side.keys() - seen_ids):
            pending_side.pop(stale_id, None)
            pending_count.pop(stale_id, None)

        now = time.time()
        delta = now - last_tick
        last_tick = now
        if delta > 0:
            instant_fps = 1.0 / delta
            fps_smoothed = instant_fps if fps_smoothed == 0 else (0.9 * fps_smoothed + 0.1 * instant_fps)

        with lock:
            system_state["cow_count"] = current_cows
            system_state["vacas_adentro"] = len(inside_ids)
            system_state["fps"] = round(fps_smoothed, 1)
            system_state["uptime_seconds"] = int(time.time() - start_time)

        annotated_frame = results[0].plot()

        if fence_overlay is not None:
            overlay_rgba(annotated_frame, fence_overlay, *fence_pos)
        else:
            cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

        cv2.putText(annotated_frame,
                    f"Adentro: {len(inside_ids)}  Ingresos: {system_state['total_ingresos']}  "
                    f"Egresos: {system_state['total_egresos']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame, jpeg_params)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.stop()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        get_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stats")
def get_stats():
    with lock:
        return system_state

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Panel Profesional - Ganado & Clima</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-main: #090d16;
                --bg-card: #111827;
                --border-color: #1f2937;
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --accent-blue: #38bdf8;
                --accent-green: #22c55e;
                --accent-orange: #f59e0b;
                --accent-red: #ef4444;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
            body { background: var(--bg-main); color: var(--text-primary); padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
            
            header { display: flex; justify-content: space-between; align-items: center; background: var(--bg-card); border: 1px solid var(--border-color); padding: 16px 24px; border-radius: 12px; }
            .logo-area h1 { font-size: 1.25rem; font-weight: 700; color: var(--accent-blue); display: flex; align-items: center; gap: 10px; }
            .logo-area span { font-size: 0.8rem; color: var(--text-secondary); font-weight: 400; }
            
            .status-badge { padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; background: rgba(56, 189, 248, 0.1); color: var(--accent-blue); border: 1px solid rgba(56, 189, 248, 0.2); }
            
            .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            @media (max-width: 1024px) { .dashboard-grid { grid-template-columns: 1fr; } }
            
            .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
            .card-title { font-size: 0.95rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between; align-items: center; }
            
            .video-wrapper { background: #000; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid var(--border-color); }
            .video-wrapper img { width: 100%; height: auto; display: block; object-fit: contain; }
            
            .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
            .metric-card { background: #0d1322; border: 1px solid var(--border-color); border-radius: 8px; padding: 14px; text-align: center; display: flex; flex-direction: column; justify-content: center; }
            .metric-card h4 { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px; }
            .metric-card .value { font-size: 1.8rem; font-weight: 700; color: var(--accent-green); }
            .metric-card.blue .value { color: var(--accent-blue); }
            .metric-card.orange .value { color: var(--accent-orange); }
            .metric-card.red .value { color: var(--accent-red); }
            
            .bottom-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            @media (max-width: 768px) { .bottom-panels { grid-template-columns: 1fr; } }
            
            .chart-container { position: relative; height: 200px; width: 100%; }
            
            .info-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color); font-size: 0.9rem; }
            .info-row span:last-child { font-weight: 600; color: var(--text-primary); }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="logo-area">
                    <h1>🐄 AgroVision IoT <span>Posadas, Misiones</span></h1>
                </div>
                <div id="status-badge" class="status-badge">Conectando...</div>
            </header>

            <div class="dashboard-grid">
                <!-- Columna de Video en Vivo -->
                <div class="card">
                    <div class="card-title">
                        <span>Cámara Principal (Corral)</span>
                        <span id="fps-display" style="color: var(--accent-blue);">0 FPS</span>
                    </div>
                    <div class="video-wrapper">
                        <img src="/video_feed" alt="Transmisión en vivo" />
                    </div>
                </div>

                <!-- Columna de Métricas Clave -->
                <div class="card">
                    <div class="card-title">Métricas de Ganado</div>
                    <div class="metrics-grid">
                        <div class="metric-card blue">
                            <h4>Adentro Corral</h4>
                            <div class="value" id="val-inside">0</div>
                        </div>
                        <div class="metric-card">
                            <h4>En Pantalla</h4>
                            <div class="value" id="val-count">0</div>
                        </div>
                        <div class="metric-card orange">
                            <h4>Total Ingresos</h4>
                            <div class="value" id="val-ingress">0</div>
                        </div>
                        <div class="metric-card red">
                            <h4>Total Egresos</h4>
                            <div class="value" id="val-egress">0</div>
                        </div>
                    </div>

                    <div class="card-title" style="margin-top: 10px;">Entorno Local</div>
                    <div class="metrics-grid">
                        <div class="metric-card blue">
                            <h4>Temperatura</h4>
                            <div class="value" id="val-temp">--°C</div>
                        </div>
                        <div class="metric-card blue">
                            <h4>Humedad</h4>
                            <div class="value" id="val-hum">--%</div>
                        </div>
                    </div>

                    <div style="margin-top: auto;">
                        <div class="info-row">
                            <span>Última actualización clima:</span>
                            <span id="val-weather-time">--</span>
                        </div>
                        <div class="info-row">
                            <span>Tiempo de actividad:</span>
                            <span id="val-uptime">0s</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Paneles Inferiores (Historial / Gráficos) -->
            <div class="bottom-panels">
                <div class="card">
                    <div class="card-title">Tendencia de Temperatura (°C)</div>
                    <div class="chart-container">
                        <canvas id="tempChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Tendencia de Humedad (%)</div>
                    <div class="chart-container">
                        <canvas id="humChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Configuración de Gráficos con Chart.js
            const maxDataPoints = 20;
            const timeLabels = [];
            const tempValues = [];
            const humValues = [];

            const tempCtx = document.getElementById('tempChart').getContext('2d');
            const tempChart = new Chart(tempCtx, {
                type: 'line',
                data: {
                    labels: timeLabels,
                    datasets: [{
                        label: 'Temp (°C)',
                        data: tempValues,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            const humCtx = document.getElementById('humChart').getContext('2d');
            const humChart = new Chart(humCtx, {
                type: 'line',
                data: {
                    labels: timeLabels,
                    datasets: [{
                        label: 'Humedad (%)',
                        data: humValues,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });

            function formatUptime(seconds) {
                const hrs = Math.floor(seconds / 3600);
                const mins = Math.floor((seconds % 3600) / 60);
                const secs = seconds % 60;
                return `${hrs}h ${mins}m ${secs}s`;
            }

            async function updateDashboard() {
                try {
                    const res = await fetch('/stats');
                    const data = await res.json();
                    
                    document.getElementById('val-count').innerText = data.cow_count;
                    document.getElementById('val-inside').innerText = data.vacas_adentro;
                    document.getElementById('val-ingress').innerText = data.total_ingresos;
                    document.getElementById('val-egress').innerText = data.total_egresos;
                    document.getElementById('fps-display').innerText = data.fps + " FPS";
                    
                    document.getElementById('val-temp').innerText = data.temperature !== "--" ? data.temperature + "°C" : "--";
                    document.getElementById('val-hum').innerText = data.humidity !== "--" ? data.humidity + "%" : "--";
                    document.getElementById('val-weather-time').innerText = data.weather_time;
                    document.getElementById('val-uptime').innerText = formatUptime(data.uptime_seconds);
                    
                    document.getElementById('status-badge').innerText = data.status;

                    // Actualizar gráficas si hay datos reales de temperatura
                    if (data.temperature !== "--") {
                        const nowStr = new Date().toLocaleTimeString();
                        if (timeLabels.length >= maxDataPoints) {
                            timeLabels.shift();
                            tempValues.shift();
                            humValues.shift();
                        }
                        timeLabels.push(nowStr);
                        tempValues.push(parseFloat(data.temperature));
                        humValues.push(parseFloat(data.humidity));
                        tempChart.update();
                        humChart.update();
                    }
                } catch (err) {
                    document.getElementById('status-badge').innerText = 'Sin conexión';
                }
            }

            setInterval(updateDashboard, 1000);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
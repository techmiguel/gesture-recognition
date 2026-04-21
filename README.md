# Gesture Recognition — ESP32-S3 + MPU6050 + TensorFlow Lite

Sistema completo de reconocimiento de gestos en tiempo real sobre hardware embebido. Un sensor IMU de 6 ejes (MPU6050) conectado a un ESP32-S3 captura movimientos, los clasifica on-device con un modelo TensorFlow Lite, y reporta los resultados por Serial USB.

**Clases reconocidas:** `arriba` · `abajo` · `izquierda` · `derecha` · `reposo`

---

## Arquitectura del sistema

```
┌─────────────────────────────── FASE 1: CAPTURA ────────────────────────────────┐
│                                                                                 │
│  [MPU6050] ──I2C──► [ESP32-S3] ──WebSocket:8765──► [capture_server.py] → CSV  │
│                      100 Hz                          ventana 1s, 50% overlap    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── FASE 2: ENTRENAMIENTO ──────────────────────────┐
│                                                                                 │
│  CSVs → [train.py] → Conv1D model → gesture_model.h + norm_params.npz         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── FASE 3: INFERENCIA ─────────────────────────────┐
│                                                                                 │
│  [MPU6050] ──I2C──► [ESP32-S3 + TFLite] ──Serial USB──► [inference_server.py]  │
│                       on-device inference              "Gesto: X | Confianza: Y" │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Componente     | Detalle                            |
|----------------|------------------------------------|
| Microcontrolador | ESP32-S3 (con PSRAM OPI 8 MB)   |
| Sensor IMU     | MPU6050 (acelerómetro + giroscopio)|
| Conexión I2C   | SDA → GPIO 8 · SCL → GPIO 9       |
| Alimentación   | 3.3 V                              |

---

## Estructura del repositorio

```
gesture_recognition/
├── firmware_esp32s3/
│   └── firmware/
│       └── esp32s3_capture/
│           ├── esp32s3_capture.ino   # Firmware de captura (streaming → PC)
│           └── secrets.h.template    # Plantilla de credenciales WiFi
│
├── esp32s3_inference/
│   ├── esp32s3_inference.ino         # Firmware de inferencia on-device (Serial)
│   └── gesture_model.h              # Modelo TFLite embebido como array C
│
├── server_python_capture_data/
│   ├── capture_server.py             # Servidor WebSocket de captura (Fase 1)
│   ├── train.py                      # Pipeline de entrenamiento (Fase 2)
│   ├── inference_server.py           # Logger Serial de inferencia (Fase 3)
│   ├── data_*.csv                    # Dataset capturado (200 muestras/clase)
│   ├── gesture_model.keras           # Modelo Keras completo
│   ├── gesture_model_float32.tflite  # Modelo TFLite (float32, 437 KB)
│   ├── norm_params.npz              # Parámetros de normalización Z-score
│   └── training_results.png         # Curvas de accuracy y matriz de confusión
│
├── .gitattributes
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Puesta en marcha

### Requisitos Python

```bash
pip install -r requirements.txt
```

### Librerías Arduino (Library Manager)

- `WebSockets` by Markus Sattler (arduinoWebSockets)
- `MPU6050` by Electronic Cats
- `TensorFlowLite_ESP32`

**Configuración de placa:** `ESP32S3 Dev Module` · PSRAM: `OPI PSRAM`

---

## Fase 1 — Captura de datos

### 1. Configurar credenciales WiFi en el firmware

```bash
cp firmware_esp32s3/firmware/esp32s3_capture/secrets.h.template \
   firmware_esp32s3/firmware/esp32s3_capture/secrets.h
# Editar secrets.h con tu SSID, contraseña e IP de la PC
```

### 2. Arrancar el servidor de captura en la PC

```bash
cd server_python_capture_data
python capture_server.py
```

### 3. Flashear y conectar el ESP32-S3

Cargar `esp32s3_capture.ino` en Arduino IDE. Al conectarse, el sensor empieza a enviar datos automáticamente.

### 4. Etiquetar gestos

Con el servidor activo, realiza el movimiento y presiona la tecla correspondiente:

| Tecla   | Etiqueta   |
|---------|------------|
| ↑       | arriba     |
| ↓       | abajo      |
| ←       | izquierda  |
| →       | derecha    |
| `SPACE` | reposo     |
| `ESC`   | salir      |

Cada pulsación guarda una ventana de 1 segundo (100 muestras) en el CSV de esa clase. Se recomienda capturar al menos **150–200 muestras por clase**.

---

## Fase 2 — Entrenamiento

```bash
cd server_python_capture_data
python train.py
```

El script genera automáticamente:

| Archivo                      | Descripción                                      |
|------------------------------|--------------------------------------------------|
| `gesture_model.keras`        | Modelo completo para inspección/fine-tuning      |
| `gesture_model_float32.tflite` | Modelo optimizado para microcontrolador        |
| `gesture_model.h`            | Array C listo para incluir en el firmware        |
| `norm_params.npz`            | Media y desviación estándar para normalización   |
| `training_results.png`       | Curvas de accuracy y matriz de confusión         |

### Arquitectura del modelo

```
Input: (100, 6)  — 100 muestras × 6 ejes IMU

Conv1D(32, kernel=5) → MaxPool(2) → Dropout(0.3)
Conv1D(64, kernel=3) → MaxPool(2) → Dropout(0.3)
Flatten → Dense(64) → Dropout(0.3)
Dense(5, softmax)
```

---

## Fase 3 — Inferencia on-device

### 1. Copiar el modelo actualizado al firmware

Tras re-entrenar, copiar el `gesture_model.h` generado a `esp32s3_inference/` y actualizar `NORM_MEAN` / `NORM_STD` en `esp32s3_inference.ino` con los valores impresos por `train.py`.

### 2. Flashear el firmware de inferencia

Cargar `esp32s3_inference.ino` en el ESP32-S3. El dispositivo clasifica gestos localmente y los imprime por Serial:

```
Gesto: Arriba | Confianza: 0.9821
```

Solo se reportan gestos con confianza ≥ **0.90**.

### 3. Leer las predicciones en la PC

Con el ESP32-S3 conectado por USB:

```bash
cd server_python_capture_data

# Detección automática de puerto
python inference_server.py

# O especificando el puerto manualmente
python inference_server.py --port COM3
```

El servidor detecta automáticamente el puerto USB, muestra las predicciones en consola y guarda un CSV de log con timestamp.

---

## Parámetros de configuración clave

| Parámetro          | Valor  | Descripción                                  |
|--------------------|--------|----------------------------------------------|
| `WINDOW_SIZE`      | 100    | Muestras por ventana (1 segundo a 100 Hz)    |
| `STEP_SIZE`        | 50     | Solapamiento entre ventanas (50%)            |
| `SAMPLE_RATE`      | 100 Hz | Frecuencia de muestreo del MPU6050           |
| `CONFIDENCE_THRESHOLD` | 0.90 | Umbral mínimo para reportar un gesto      |
| `TENSOR_ARENA_SIZE`| 300 KB | Memoria PSRAM reservada para TFLite         |

---

## Datos de entrenamiento incluidos

El repositorio incluye el dataset capturado (~200 muestras por clase, 1000 muestras en total):

| Clase      | Muestras |
|------------|----------|
| arriba     | 201      |
| abajo      | 200      |
| izquierda  | 200      |
| derecha    | 200      |
| reposo     | 200      |

---

## Licencia

MIT — ver [LICENSE](LICENSE).

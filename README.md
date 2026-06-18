[![Platform](https://img.shields.io/badge/Platform-ESP32--S3-blue)](https://www.espressif.com/en/products/socs/esp32-s3)
[![Framework](https://img.shields.io/badge/Framework-TFLite--Micro-green)](https://github.com/tensorflow/tflite-micro)
[![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20Python-orange)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red)](YOUR_LINK_HERE)

# Gesture Recognition — ESP32-S3 + MPU6050 + TFLite Micro

5-class directional gesture classifier running fully on-device on an ESP32-S3: a two-block Conv1D model (~110K parameters, float32, 437 KB) classifies 1-second windows of 6-axis MPU6050 data at 1 Hz with **98.5% test accuracy on 201 held-out samples**, deploying via TFLite Micro with a 300 KB tensor arena allocated in OPI PSRAM using `ps_malloc()` — required because the model overflows the ESP32-S3's internal DRAM heap.

The pipeline is fully self-contained: a dedicated capture firmware streams 100 Hz IMU data over WebSocket to a Python labeling server; `train.py` trains, evaluates, and exports the model as a C byte array; the inference firmware embeds that array, applies hardcoded Z-score normalization, and writes float32 values directly into `input->data.f[]` — a design constraint that drives the deliberate choice of float32 over int8 quantization.

---

## Hardware & System Overview

| Component | Part | Role |
|-----------|------|------|
| MCU | ESP32-S3 DevKit (OPI PSRAM 8 MB) | On-device inference, WiFi (capture phase), I2C master |
| IMU | MPU6050 | 6-axis inertial sensing: accelerometer ±2g + gyroscope ±250°/s |
| Bus | I2C Fast Mode (400 kHz) | SDA → GPIO 8 · SCL → GPIO 9 |
| Power | 3.3 V | — |

### System architecture

```
╔═══════════════════════════ PHASE 1 — DATA CAPTURE ════════════════════════════╗
║                                                                                ║
║  [MPU6050] ──I2C 400kHz──► [ESP32-S3]  ──WiFi WebSocket :8765──► [capture_server.py]
║   ±2g / ±250°/s             100 Hz, 6 axes                        deque(maxlen=100)
║                              micros()-timed                        50% overlap
║                                                                    keypress → CSV
╚════════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════ PHASE 2 — TRAINING ═════════════════════════════════╗
║                                                                                ║
║  data_*.csv ──► [train.py] ──► gesture_model.h  (C array, 437 KB)             ║
║                  Z-score norm    norm_params.npz  (mean/std → hardcoded in fw) ║
║                  Conv1D ×2       gesture_model_float32.tflite                  ║
║                  TFLite float32 export                                         ║
╚════════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════ PHASE 3 — ON-DEVICE INFERENCE ══════════════════════╗
║                                                                                ║
║  [MPU6050] ──I2C──► [ESP32-S3 TFLite Micro] ──Serial 115200──► [inference_server.py]
║   100 Hz              300 KB tensor arena in PSRAM              CSV log + console
║                        ps_malloc(), Invoke() @ 1 Hz             conf ≥ 0.90 threshold
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## Dataset

### Gesture classes

| Index | Class | Direction | CSV file | Test samples |
|-------|-------|-----------|----------|-------------|
| 0 | abajo | Down sweep | `data_abajo.csv` | 40 |
| 1 | arriba | Up sweep | `data_arriba.csv` | 41 |
| 2 | derecha | Right sweep | `data_derecha.csv` | 40 |
| 3 | izquierda | Left sweep | `data_izquierda.csv` | 40 |
| 4 | reposo | Rest / idle | `data_reposo.csv` | 40 |

Index ordering is set by `sklearn.LabelEncoder` with `fit(["arriba","abajo","izquierda","derecha","reposo"])`, which sorts alphabetically. The identical ordering is hardcoded in `GESTURE_NAMES[5]` in `esp32s3_inference.ino`.

### Sample counts

| Class | Total samples |
|-------|--------------|
| arriba | 201 |
| abajo | 200 |
| derecha | 200 |
| izquierda | 200 |
| reposo | 200 |
| **Total** | **1001** |

Split: `TEST_SIZE = 0.20`, `VAL_SIZE = 0.15` (of the training portion), stratified by class, `random_state=42`.

### Data format

Each CSV row is a flattened sliding window: 100 timesteps × 6 axes = **600 float columns**, named `ax_0 … ax_99, ay_0 … ay_99, …, gz_0 … gz_99`. Values are in physical units — accelerometer in **g** (`ACC_SCALE = 1/16384`), gyroscope in **°/s** (`GYR_SCALE = 1/131`).

### Collection procedure (`capture_server.py` + `esp32s3_capture.ino`)

`esp32s3_capture.ino` samples the MPU6050 at 100 Hz using `micros()`-based timing (not `delay()`), converts raw 16-bit ADC counts to physical units in a single I2C transaction (`getMotion6()`), and streams comma-separated readings over WebSocket to port 8765.

`capture_server.py` maintains a circular `deque(maxlen=100)` buffer. When the operator performs a gesture and presses an arrow key or spacebar, it flushes the current 100-sample window to the matching CSV (`data_<label>.csv`) and advances the read pointer by 50 samples (50% overlap via `popleft()` × 50). Overlap doubles effective sample count per capture session without additional physical repetitions.

---

## Model Architecture

### Layer stack

```
Input      shape=(100, 6)   — 100 timesteps × 6 IMU axes (ax, ay, az, gx, gy, gz)

Conv1D     filters=32, kernel_size=5, padding=same, activation=relu  → (100, 32)
MaxPool1D  pool_size=2                                                → (50,  32)
Dropout    rate=0.3

Conv1D     filters=64, kernel_size=3, padding=same, activation=relu  → (50,  64)
MaxPool1D  pool_size=2                                                → (25,  64)
Dropout    rate=0.3

Flatten                                                               → (1600,)
Dense      units=64, activation=relu                                  → (64,)
Dropout    rate=0.3

Dense      units=5,  activation=softmax                               → (5,)   ← output
```

### Parameter count

| Layer | Calculation | Parameters |
|-------|-------------|-----------|
| Conv1D(32, k=5) | 5 × 6 × 32 + 32 | 992 |
| Conv1D(64, k=3) | 3 × 32 × 64 + 64 | 6,208 |
| Dense(64) | 1600 × 64 + 64 | 102,464 |
| Dense(5) | 64 × 5 + 5 | 325 |
| **Total** | | **109,989** |

The `Dense(64)` layer accounts for 93% of all parameters (the 1600-unit flattened output of `MaxPool1D` is the bottleneck). Applying global average pooling instead of `Flatten` before `Dense` would reduce this substantially at the cost of some accuracy.

### Why Conv1D rather than MLP or Conv2D

An MLP would flatten the `(100, 6)` input to a 600-element vector immediately, discarding temporal order — the distinguishing feature of a directional sweep is the *sequence* of accelerations, not their histogram. Conv2D would impose spatial adjacency across the 6 sensor axes, which have no geometric relationship. Conv1D slides a 1D kernel along the time axis: a kernel of size 5 at 100 Hz covers 50 ms and learns local motion patterns (onset, peak, deceleration of a sweep) regardless of their position within the window.

---

## Training Pipeline

**Script:** `server_python_capture_data/train.py`

1. **Load**: reads five `data_<label>.csv` files, concatenates, warns if class imbalance ratio exceeds 1.5×.
2. **Reshape**: each CSV row (600 floats) is reshaped to `(100, 6)` → dataset shape `(N, 100, 6)`.
3. **Label encode**: `sklearn.LabelEncoder.fit(["arriba","abajo","izquierda","derecha","reposo"])` — alphabetical sort determines final integer labels.
4. **Stratified split**: `train_test_split(test_size=0.20, stratify=y, random_state=42)`.
5. **Z-score normalization**: mean and std computed from `X_train` only (axes 0 and 1, i.e., per-feature across all train windows). Saved to `norm_params.npz`. These six mean and std values are hardcoded verbatim in `esp32s3_inference.ino` as `NORM_MEAN[6]` and `NORM_STD[6]`.
6. **Train**: Adam optimizer, `sparse_categorical_crossentropy`, `batch_size=32`, up to `EPOCHS=100`. Callbacks: `EarlyStopping(monitor=val_loss, patience=12, restore_best_weights=True)` and `ReduceLROnPlateau(patience=6, factor=0.5, min_lr=1e-6)`.
7. **Evaluate**: computes per-class `classification_report` and saves `training_results.png` (accuracy + validation curves, confusion matrix).
8. **Export TFLite (float32)**: `TFLiteConverter.from_keras_model()` with no quantization directives. Output: `gesture_model_float32.tflite` (437 KB).
9. **Generate C header**: serializes the `.tflite` bytes as a hex array into `gesture_model.h`. The variable is named `gesture_model_int8_tflite` — a legacy name preserved for compatibility with an earlier EloquentTinyML-based revision of the firmware. The model content is float32.

### Generated outputs

| File | Description |
|------|-------------|
| `gesture_model.keras` | Full Keras model for inspection or fine-tuning |
| `gesture_model_float32.tflite` | TFLite graph, float32, no quantization, 437 KB |
| `gesture_model.h` | `const uint8_t gesture_model_int8_tflite[]` — embed in firmware |
| `norm_params.npz` | Per-axis Z-score parameters (copy values into firmware manually) |
| `training_results.png` | Accuracy/loss curves and confusion matrix |

---

## Deployment

### Embedding the model in firmware

`gesture_model.h` declares an `alignas(8) const uint8_t gesture_model_int8_tflite[]` array that is linked directly into the firmware image. TFLite Micro parses it at runtime with `tflite::GetModel(gesture_model_int8_tflite)` — no filesystem (SPIFFS/LittleFS) or SD card is required. Array alignment to 8 bytes satisfies TFLite Micro's internal alignment requirements.

### TFLite Micro initialization (`esp32s3_inference.ino`)

The firmware uses the `TensorFlowLite_ESP32` library (TFLite Micro). A `MicroMutableOpResolver<9>` registers exactly the ops the exported graph requires:

```cpp
op_resolver->AddConv2D();       // Conv1D is lowered to ExpandDims + Conv2D internally
op_resolver->AddMaxPool2D();    // MaxPool1D → MaxPool2D, same reason
op_resolver->AddFullyConnected();
op_resolver->AddSoftmax();
op_resolver->AddReshape();
op_resolver->AddExpandDims();   // ┐
op_resolver->AddShape();        // │ required for the Conv1D→Conv2D lowering
op_resolver->AddStridedSlice(); // │
op_resolver->AddPack();         // ┘
```

TFLite Micro has no native Conv1D kernel; it lowers `Conv1D → ExpandDims + Conv2D + Squeeze`, requiring the auxiliary shape-manipulation ops. Registering only the ops actually used (rather than `AllOpsResolver`) keeps the firmware binary smaller.

All TFLite objects (`MicroErrorReporter`, `MicroMutableOpResolver`, `MicroInterpreter`) are allocated on the heap inside `setup()` rather than as global-scope constructors, avoiding C++ static initialization-order issues at boot. The loop task stack is raised to 32 KB via `SET_LOOP_TASK_STACK_SIZE(32 * 1024)` to accommodate TFLite Micro's operator call depth during `Invoke()`.

### Why float32 and not int8

Post-training int8 quantization would set the input tensor type to `kTfLiteInt8`, requiring the firmware to apply per-channel quantization scale and zero-point before writing to `input->data.int8[]`. The inference firmware instead normalizes and writes directly:

```cpp
input->data.f[i * NUM_AXES + j] = (window_buf[i][j] - NORM_MEAN[j]) / NORM_STD[j];
```

This path is only valid when `input->type == kTfLiteFloat32`. Keeping the model in float32 eliminates the quantization preprocessing step, avoids any accuracy loss from weight quantization on a small (1001-sample) dataset, and keeps the inference code straightforward. The trade-off is a larger model (437 KB vs. ~110 KB for int8).

### Why `ps_malloc()` is required for the tensor arena

`TENSOR_ARENA_SIZE = 300 * 1024 = 307,200 bytes`. After FreeRTOS, the Arduino runtime, and the loop task stack, the ESP32-S3's internal DRAM free heap is typically under 250 KB — insufficient for the arena alone. `ps_malloc()` allocates from the 8 MB OPI PSRAM, which is accessible as heap when the board is configured with **PSRAM: OPI PSRAM** in Arduino IDE. The firmware checks the return value immediately:

```cpp
tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
if (!tensor_arena) {
    Serial.println("ERROR: ps_malloc falló");
    while (true) { delay(1000); }
}
```

A `nullptr` return (PSRAM not enabled or not present) causes an immediate halt with a Serial error, rather than a silent crash later inside `AllocateTensors()`.

---

## Results

### Test accuracy

**98.5%** — 198 correct out of 201 test samples (20% stratified holdout, `random_state=42`).

### Per-class metrics (derived from confusion matrix)

| Class | Precision | Recall | F1-score | Test samples |
|-------|-----------|--------|----------|-------------|
| abajo | 95.1% | 97.5% | 96.3% | 40 |
| arriba | 97.5% | 95.1% | 96.3% | 41 |
| derecha | 100% | 100% | 100% | 40 |
| izquierda | 100% | 100% | 100% | 40 |
| reposo | 100% | 100% | 100% | 40 |

### Confusion matrix

```
              Predicted →
              abajo  arriba  derecha  izquierda  reposo
Actual abajo     39       1        0          0       0
       arriba     2      39        0          0       0
      derecha     0       0       40          0       0
    izquierda     0       0        0         40       0
       reposo     0       0        0          0      40
```

All 3 misclassifications occur between `abajo` and `arriba` only. `derecha`, `izquierda`, and `reposo` achieve perfect separation. Training converged in approximately 28 epochs before `EarlyStopping` triggered. Accuracy and validation curves are in `server_python_capture_data/training_results.png`.

---

## Usage

### Prerequisites

**Python:**
```bash
pip install -r requirements.txt
# tensorflow>=2.13, numpy, pandas, scikit-learn, matplotlib, websockets, pynput, pyserial
```

**Arduino IDE libraries** (Tools → Manage Libraries):
- `WebSockets` by Markus Sattler (arduinoWebSockets)
- `MPU6050` by Electronic Cats
- `TensorFlowLite_ESP32`

**Arduino IDE board settings** (Tools menu):

| Setting | Value |
|---------|-------|
| Board | ESP32S3 Dev Module |
| PSRAM | OPI PSRAM |

**Wiring:**

| MPU6050 | ESP32-S3 |
|---------|---------|
| SDA | GPIO 8 |
| SCL | GPIO 9 |
| VCC | 3.3 V |
| GND | GND |

---

### Phase 1 — Capture training data  *(skip if using the included dataset)*

**1. Configure WiFi credentials:**

```bash
cp firmware_esp32s3/firmware/esp32s3_capture/secrets.h.template \
   firmware_esp32s3/firmware/esp32s3_capture/secrets.h
# Edit secrets.h: fill SECRET_WIFI_SSID, SECRET_WIFI_PASSWORD, SECRET_SERVER_IP
# (SECRET_SERVER_IP is the local IP of the PC running capture_server.py)
```

**2. Start the capture server on the PC:**

```bash
cd server_python_capture_data
python capture_server.py
```

**3. Flash and connect the ESP32-S3:**

Open `firmware_esp32s3/firmware/esp32s3_capture/esp32s3_capture.ino` in Arduino IDE and flash. Monitor Serial at 115200 baud — a `[WS] Conectado` message confirms the WebSocket link.

**4. Label gestures:**

Perform a gesture, then immediately press the matching key:

| Key | Gesture |
|-----|---------|
| ↑ | arriba (up) |
| ↓ | abajo (down) |
| ← | izquierda (left) |
| → | derecha (right) |
| `Space` | reposo (rest/idle) |
| `Esc` | quit and flush CSV files |

Each keypress saves one 1-second window (100 samples) with 50% overlap. Aim for ≥ 150 samples per class before training.

---

### Phase 2 — Train the model

```bash
cd server_python_capture_data
python train.py
```

When training completes, the script prints the normalization parameters:

```
  ax: mean=0.646844  std=0.296141
  ay: mean=0.202382  std=0.310030
  ...
```

Copy `gesture_model.h` to `esp32s3_inference/` and update `NORM_MEAN[6]` and `NORM_STD[6]` in `esp32s3_inference.ino` with the printed values.

---

### Phase 3 — Run on-device inference

**1. Flash the inference firmware:**

Open `esp32s3_inference/esp32s3_inference.ino` in Arduino IDE and flash. Successful boot prints:

```
=== BOOT ===
MPU6050 OK
Modelo cargado OK
```

If `ps_malloc` fails, Serial will print `ERROR: ps_malloc falló` — verify that PSRAM is set to `OPI PSRAM` in board settings.

**2. Read predictions on the PC:**

```bash
cd server_python_capture_data

# Auto-detect COM port (looks for Silicon Labs / CH340 / CP210x descriptors)
python inference_server.py

# Or specify manually
python inference_server.py --port COM3         # Windows
python inference_server.py --port /dev/ttyUSB0 # Linux/macOS
```

The firmware outputs only predictions with confidence ≥ 0.90:

```
Gesto: Arriba | Confianza: 0.9821
```

`inference_server.py` displays a per-class confidence bar with running counts and saves a timestamped log to `inference_log_YYYYMMDD_HHMMSS.csv`.

---

## Roadmap

- [ ] Measure and document on-device inference latency (currently untracked in firmware)
- [ ] Evaluate int8 quantization with explicit `input->data.int8[]` handling to reduce model flash footprint from 437 KB to ~110 KB
- [ ] Extend dataset to 8+ gesture classes
- [ ] Port firmware to PlatformIO for reproducible dependency pinning
- [ ] Replace USB Serial output with BLE notifications for untethered operation

---

## License

MIT — see [LICENSE](LICENSE).

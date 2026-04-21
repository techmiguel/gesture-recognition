# =============================================================================
# Servidor de Captura de Gestos — WebSocket → CSV por etiqueta
#
# Etiquetas disponibles:
#   ↑  → arriba      ↓  → abajo
#   ←  → izquierda   →  → derecha
#   SPACE → reposo   ESC → salir
#
# Cada etiqueta genera y escribe en su propio archivo CSV:
#   data_arriba.csv, data_abajo.csv, data_izquierda.csv,
#   data_derecha.csv, data_reposo.csv
#
# Dependencias:
#   pip install websockets pynput
#
# Uso:
#   python capture_server.py
#   (Ejecutar ANTES de encender el ESP32-S3)
# =============================================================================

import asyncio
import websockets
import csv
import os
from datetime import datetime
from pynput import keyboard
from collections import deque

# ── Configuración ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 100
WINDOW_SIZE  = 100   # muestras = 1 segundo a 100 Hz
STEP_SIZE    = 50    # 50% solapamiento

LABELS = {
    keyboard.Key.up:    "arriba",
    keyboard.Key.down:  "abajo",
    keyboard.Key.left:  "izquierda",
    keyboard.Key.right: "derecha",
    keyboard.Key.space: "reposo",
}

NUM_AXES = 6
AXES     = ["ax", "ay", "az", "gx", "gy", "gz"]

HEADER = [f"{ax}_{i}" for i in range(WINDOW_SIZE) for ax in AXES]

# ── Estado global ─────────────────────────────────────────────────────────────
buffer        = deque(maxlen=WINDOW_SIZE)
pending_label = None

# Contadores por etiqueta para feedback en consola
counters = {name: 0 for name in LABELS.values()}

# Escritores CSV — uno por etiqueta, abiertos en modo append
csv_writers  = {}
csv_files    = {}

# ── Gestión de archivos CSV ───────────────────────────────────────────────────
def get_writer(label: str):
    """Devuelve el csv.writer para la etiqueta dada, creándolo si no existe."""
    if label in csv_writers:
        return csv_writers[label]

    filepath = f"data_{label}.csv"
    file_exists = os.path.isfile(filepath)

    f = open(filepath, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(HEADER)
        print(f"[CSV] Creado: {filepath}")
    else:
        # Contar muestras existentes para mostrar en el arranque
        with open(filepath, "r") as tmp:
            existing = sum(1 for _ in tmp) - 1  # restar header
        counters[label] = max(existing, 0)
        print(f"[CSV] Abierto (existente): {filepath} — {counters[label]} muestras previas")

    csv_files[label]   = f
    csv_writers[label] = writer
    return writer

def save_window(label: str):
    if len(buffer) < WINDOW_SIZE:
        print(f"[WARN] Buffer incompleto ({len(buffer)}/{WINDOW_SIZE}) — descartado")
        return

    row = [v for sample in list(buffer) for v in sample]
    writer = get_writer(label)
    writer.writerow(row)
    csv_files[label].flush()   # escribir a disco inmediatamente

    counters[label] += 1
    total = sum(counters.values())
    print(f"[SAVE] {label:<12} #{counters[label]:>3}   |   "
          + "  ".join(f"{k}: {v}" for k, v in counters.items())
          + f"   |   total: {total}")

    # Avanzar buffer para solapamiento
    for _ in range(STEP_SIZE):
        if buffer:
            buffer.popleft()

def close_all():
    for label, f in csv_files.items():
        f.close()
    print("[CSV] Todos los archivos cerrados.")

# ── Teclado ───────────────────────────────────────────────────────────────────
def on_key_press(key):
    global pending_label
    if key in LABELS:
        pending_label = LABELS[key]
        print(f"[KEY] Etiqueta pendiente → '{pending_label}'")
    elif key == keyboard.Key.esc:
        print("\n[EXIT] ESC recibido — cerrando servidor...")
        close_all()
        os._exit(0)

# ── WebSocket ─────────────────────────────────────────────────────────────────
async def ws_handler(websocket):
    global pending_label
    print(f"[CONN] ESP32-S3 conectado desde {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                values = tuple(float(x) for x in message.strip().split(","))
                if len(values) != NUM_AXES:
                    print(f"[WARN] Se esperaban {NUM_AXES} valores, se recibieron {len(values)}")
                    continue

                buffer.append(values)

                if pending_label and len(buffer) == WINDOW_SIZE:
                    save_window(pending_label)
                    pending_label = None

            except ValueError:
                print(f"[WARN] Mensaje malformado: '{message}'")

    except websockets.exceptions.ConnectionClosed:
        print("[CONN] ESP32-S3 desconectado")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    # Pre-abrir todos los CSV al arranque para mostrar estado inicial
    for label in LABELS.values():
        get_writer(label)

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    print()
    print("=" * 60)
    print("  Servidor de Captura de Gestos — WebSocket :8765")
    print("=" * 60)
    print("  ↑  arriba      ↓  abajo")
    print("  ←  izquierda   →  derecha")
    print("  SPACE  reposo  |  ESC  salir")
    print("=" * 60)
    print(f"  Ventana: {WINDOW_SIZE} muestras @ {SAMPLE_RATE} Hz = {WINDOW_SIZE/SAMPLE_RATE:.1f} s")
    print(f"  Solapamiento: {STEP_SIZE} muestras (50%)")
    print("=" * 60)
    print()

    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

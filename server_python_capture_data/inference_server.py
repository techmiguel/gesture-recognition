# =============================================================================
# Servidor de Inferencia — Recibe predicciones del ESP32-S3 vía WebSocket
#
# El ESP32-S3 envía JSON con el formato:
#   {"gesture":"arriba","confidence":0.9821,"timestamp":12345}
#
# Este servidor:
#   - Muestra cada predicción en consola con formato legible
#   - Lleva conteo de gestos detectados por clase
#   - Guarda un log CSV con timestamp real de PC + datos recibidos
#
# Dependencias:
#   pip install websockets
#
# Uso:
#   python inference_server.py
#   (Ejecutar ANTES de encender o resetear el ESP32-S3)
# =============================================================================

import asyncio
import websockets
import json
import csv
import os
from datetime import datetime
from collections import defaultdict

# ── Configuración ─────────────────────────────────────────────────────────────
HOST      = "0.0.0.0"
PORT      = 8766
LOG_FILE  = f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

CSV_HEADER = ["pc_timestamp", "gesture", "confidence", "esp32_timestamp_ms"]

# ── Estado ────────────────────────────────────────────────────────────────────
counters = defaultdict(int)
log_file  = open(LOG_FILE, "w", newline="")
writer    = csv.writer(log_file)
writer.writerow(CSV_HEADER)
log_file.flush()

# Iconos por gesto para feedback visual rápido en consola
ICONS = {
    "arriba":    "↑",
    "abajo":     "↓",
    "izquierda": "←",
    "derecha":   "→",
    "reposo":    "·",
}

# ── WebSocket handler ─────────────────────────────────────────────────────────
async def ws_handler(websocket):
    print(f"[CONN] ESP32-S3 conectado desde {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                gesture    = data.get("gesture",    "desconocido")
                confidence = data.get("confidence", 0.0)
                esp_ts     = data.get("timestamp",  0)
                pc_ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Actualizar contadores
                counters[gesture] += 1
                total = sum(counters.values())

                # Consola
                icon = ICONS.get(gesture, "?")
                bar  = "█" * int(confidence * 20)  # barra de confianza 20 chars
                print(f"  {icon}  {gesture:<12} {confidence:.4f}  {bar:<20}  "
                      f"#{counters[gesture]:>4}  |  total: {total}")

                # Log CSV
                writer.writerow([pc_ts, gesture, f"{confidence:.6f}", esp_ts])
                log_file.flush()

            except json.JSONDecodeError:
                print(f"[WARN] Mensaje no es JSON válido: '{message}'")

    except websockets.exceptions.ConnectionClosed:
        print("[CONN] ESP32-S3 desconectado")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print()
    print("=" * 60)
    print("  Servidor de Inferencia de Gestos — WebSocket :8766")
    print("=" * 60)
    print(f"  Log guardado en: {LOG_FILE}")
    print(f"  Esperando conexión del ESP32-S3...")
    print()
    print(f"  {'GESTO':<14} {'CONF':>6}  {'BARRA':<20}  {'#':>5}  TOTAL")
    print("  " + "-" * 56)

    async with websockets.serve(ws_handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Servidor detenido.")
        print("\n  Resumen de sesión:")
        for gesture, count in sorted(counters.items()):
            print(f"    {gesture:<12}: {count}")
        log_file.close()
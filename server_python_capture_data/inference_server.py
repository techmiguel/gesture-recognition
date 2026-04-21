# =============================================================================
# Servidor de Inferencia — Lee predicciones del ESP32-S3 vía puerto Serial
#
# El ESP32-S3 imprime líneas con el formato:
#   Gesto: arriba | Confianza: 0.9821
#
# Este servidor:
#   - Muestra cada predicción en consola con formato legible
#   - Lleva conteo de gestos detectados por clase
#   - Guarda un log CSV con timestamp real de PC + datos recibidos
#
# Dependencias:
#   pip install pyserial
#
# Uso:
#   python inference_server.py [--port COM3] [--baud 115200]
#   (El ESP32-S3 debe estar conectado por USB antes de ejecutar)
# =============================================================================

import serial
import serial.tools.list_ports
import csv
import re
import argparse
import sys
from datetime import datetime
from collections import defaultdict

# ── Configuración ─────────────────────────────────────────────────────────────
DEFAULT_BAUD = 115200
LOG_FILE     = f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
CSV_HEADER   = ["pc_timestamp", "gesture", "confidence"]

# Patrón de línea del firmware:  "Gesto: Arriba | Confianza: 0.9821"
LINE_PATTERN = re.compile(
    r"Gesto:\s*(\w+)\s*\|\s*Confianza:\s*([0-9.]+)",
    re.IGNORECASE
)

ICONS = {
    "arriba":    "↑",
    "abajo":     "↓",
    "izquierda": "←",
    "derecha":   "→",
    "reposo":    "·",
}

# ── Detección automática de puerto ───────────────────────────────────────────
def find_esp32_port() -> str | None:
    """Devuelve el primer puerto con descripción de chip Silicon Labs o CH340."""
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(kw in desc for kw in ("silicon labs", "ch340", "cp210", "esp32", "uart")):
            return p.device
    return None

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inference logger — ESP32-S3 Serial")
    parser.add_argument("--port", default=None,
                        help="Puerto Serial (ej. COM3, /dev/ttyUSB0). "
                             "Si se omite, se detecta automáticamente.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help=f"Baudrate (default: {DEFAULT_BAUD})")
    args = parser.parse_args()

    port = args.port or find_esp32_port()
    if not port:
        print("[ERROR] No se encontró ningún ESP32 conectado por USB.")
        print("        Conecta el dispositivo o especifica el puerto con --port COMx")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Servidor de Inferencia de Gestos — Serial")
    print("=" * 60)
    print(f"  Puerto : {port}  |  Baudrate: {args.baud}")
    print(f"  Log    : {LOG_FILE}")
    print()
    print(f"  {'GESTO':<14} {'CONF':>6}  {'BARRA':<20}  {'#':>5}  TOTAL")
    print("  " + "-" * 56)

    counters = defaultdict(int)

    with open(LOG_FILE, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(CSV_HEADER)
        log_file.flush()

        try:
            with serial.Serial(port, args.baud, timeout=2) as ser:
                print(f"[CONN] Conectado a {port}\n")
                while True:
                    try:
                        raw = ser.readline()
                    except serial.SerialException as e:
                        print(f"\n[ERROR] Conexión perdida: {e}")
                        break

                    if not raw:
                        continue

                    try:
                        line = raw.decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue

                    m = LINE_PATTERN.search(line)
                    if not m:
                        continue

                    gesture    = m.group(1).lower()
                    confidence = float(m.group(2))
                    pc_ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    counters[gesture] += 1
                    total = sum(counters.values())

                    icon = ICONS.get(gesture, "?")
                    bar  = "█" * int(confidence * 20)
                    print(f"  {icon}  {gesture:<12} {confidence:.4f}  {bar:<20}  "
                          f"#{counters[gesture]:>4}  |  total: {total}")

                    writer.writerow([pc_ts, gesture, f"{confidence:.6f}"])
                    log_file.flush()

        except serial.SerialException as e:
            print(f"\n[ERROR] No se pudo abrir {port}: {e}")
            print("        Verifica que el puerto es correcto y no está en uso.")
            sys.exit(1)
        except KeyboardInterrupt:
            pass

    print(f"\n[EXIT] Sesión terminada.")
    print("\n  Resumen:")
    for gesture, count in sorted(counters.items()):
        print(f"    {gesture:<12}: {count}")
    print(f"\n  Log guardado en: {LOG_FILE}")


if __name__ == "__main__":
    main()

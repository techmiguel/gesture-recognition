// =============================================================================
// ESP32-S3 — Firmware de Captura de Gestos (MPU6050 → WebSocket → PC)
// Modo: CAPTURA (streaming de datos en crudo al servidor Python)
//
// Hardware:
//   - Microcontrolador : ESP32-S3
//   - Sensor           : MPU6050
//   - I2C SDA          : GPIO 8
//   - I2C SCL          : GPIO 9
//
// Dependencias (instalar desde Arduino IDE > Library Manager):
//   - "WebSockets" by Markus Sattler      (arduinoWebSockets)
//   - "MPU6050" by Electronic Cats        (ElectronicCats/MPU6050)
//
// Placa en Arduino IDE:
//   Tools > Board > ESP32S3 Dev Module
//   Tools > PSRAM > OPI PSRAM (si tienes 8MB PSRAM)
//
// Uso:
//   1. Ajustar WIFI_SSID, WIFI_PASSWORD y SERVER_IP abajo.
//   2. Cargar este sketch en el ESP32-S3.
//   3. Abrir Serial Monitor a 115200 para ver logs.
//   4. Ejecutar capture_server.py en la PC ANTES de encender el ESP32-S3.
// =============================================================================

#include <Wire.h>
#include <WiFi.h>
#include <WebSocketsClient.h>   // arduinoWebSockets
#include <MPU6050.h>            // ElectronicCats/MPU6050

// -----------------------------------------------------------------------------
// CONFIGURACIÓN — Copiar secrets.h.template como secrets.h y rellenar valores
// -----------------------------------------------------------------------------
#include "secrets.h"

static const char*    WIFI_SSID       = SECRET_WIFI_SSID;
static const char*    WIFI_PASSWORD   = SECRET_WIFI_PASSWORD;
static const char*    SERVER_IP       = SECRET_SERVER_IP;   // IP local de la PC
static const uint16_t SERVER_PORT     = 8765;
static const char*    WS_PATH         = "/";

// Muestreo
static const int      SAMPLE_RATE_HZ  = 100;             // frecuencia objetivo
static const uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

// Pines I2C
static const int      PIN_SDA         = 8;
static const int      PIN_SCL         = 9;

// Factores de conversión MPU6050
//   Acelerómetro ±2g  → 16384 LSB/g
//   Giroscopio   ±250°/s → 131 LSB/(°/s)
static const float    ACC_SCALE       = 1.0f / 16384.0f;
static const float    GYR_SCALE       = 1.0f / 131.0f;

// -----------------------------------------------------------------------------
// Variables globales
// -----------------------------------------------------------------------------
MPU6050          mpu;
WebSocketsClient ws;

volatile bool    wsConnected    = false;
uint32_t         lastSampleUs   = 0;
uint32_t         sampleCount    = 0;
uint32_t         lastStatsMs    = 0;

// Buffer de mensaje: "±X.XXXX,±X.XXXX,±X.XXXX,±XXX.XX,±XXX.XX,±XXX.XX\0"
// Máximo ~50 caracteres; 128 es conservador
char             msgBuf[128];

// -----------------------------------------------------------------------------
// Callback WebSocket
// -----------------------------------------------------------------------------
void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {

        case WStype_CONNECTED:
            wsConnected = true;
            Serial.printf("[WS] Conectado a ws://%s:%u%s\n",
                          SERVER_IP, SERVER_PORT, WS_PATH);
            break;

        case WStype_DISCONNECTED:
            wsConnected = false;
            Serial.println("[WS] Desconectado — reintentando...");
            break;

        case WStype_TEXT:
            // El servidor puede enviar mensajes de control; los ignoramos
            Serial.printf("[WS] Servidor: %s\n", payload);
            break;

        case WStype_ERROR:
            Serial.printf("[WS] Error: %s\n", payload ? (char*)payload : "desconocido");
            break;

        default:
            break;
    }
}

// -----------------------------------------------------------------------------
// Inicialización WiFi con reintentos
// -----------------------------------------------------------------------------
void initWiFi() {
    Serial.printf("[WiFi] Conectando a '%s'", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint8_t attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        if (++attempts >= 40) {   // 20 segundos máximo
            Serial.println("\n[WiFi] ERROR: No se pudo conectar. Reiniciando...");
            delay(1000);
            ESP.restart();
        }
    }
    Serial.printf("\n[WiFi] Conectado. IP local: %s\n",
                  WiFi.localIP().toString().c_str());
}

// -----------------------------------------------------------------------------
// Inicialización MPU6050
// -----------------------------------------------------------------------------
void initMPU() {
    Wire.begin(PIN_SDA, PIN_SCL);
    Wire.setClock(400000);   // I2C Fast Mode (400 kHz)

    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("[MPU] ERROR: MPU6050 no responde en I2C.");
        Serial.println("      Verificar cableado SDA→GPIO8, SCL→GPIO9, VCC→3.3V, GND.");
        while (true) {
            delay(1000);
        }
    }

    // Configurar rangos de máxima resolución
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);    // ±2g
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);    // ±250 °/s

    // Filtro paso-bajo digital: reducir ruido de alta frecuencia
    // DLPF_CFG = 3 → BW ≈ 44 Hz (adecuado para movimientos manuales)
    mpu.setDLPFMode(MPU6050_DLPF_BW_42);

    Serial.println("[MPU] MPU6050 inicializado correctamente.");
    Serial.printf("      Acelerómetro: ±2g  | Escala: %.6f g/LSB\n", ACC_SCALE);
    Serial.printf("      Giroscopio  : ±250°/s | Escala: %.6f (°/s)/LSB\n", GYR_SCALE);
}

// -----------------------------------------------------------------------------
// setup()
// -----------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(500);   // tiempo para que el monitor serial se conecte

    Serial.println("==============================================");
    Serial.println("  ESP32-S3 | Captura de Gestos v1.0");
    Serial.println("==============================================");

    initMPU();
    initWiFi();

    ws.begin(SERVER_IP, SERVER_PORT, WS_PATH);
    ws.onEvent(onWebSocketEvent);
    ws.setReconnectInterval(3000);   // reintentar cada 3 s si se pierde conexión

    lastSampleUs = micros();
    lastStatsMs  = millis();

    Serial.printf("[INIT] Frecuencia de muestreo objetivo: %d Hz\n", SAMPLE_RATE_HZ);
    Serial.println("[INIT] Esperando conexión WebSocket...");
}

// -----------------------------------------------------------------------------
// loop()
// -----------------------------------------------------------------------------
void loop() {
    // Mantener la conexión WebSocket activa (reconexión automática incluida)
    ws.loop();

    // Temporización por micros() para mayor precisión que delay()
    uint32_t now = micros();
    if ((now - lastSampleUs) >= SAMPLE_INTERVAL_US) {
        lastSampleUs = now;

        // Leer los 6 ejes en una sola transacción I2C
        int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
        mpu.getMotion6(&ax_raw, &ay_raw, &az_raw,
                       &gx_raw, &gy_raw, &gz_raw);

        // Convertir a unidades físicas
        float ax = ax_raw * ACC_SCALE;
        float ay = ay_raw * ACC_SCALE;
        float az = az_raw * ACC_SCALE;
        float gx = gx_raw * GYR_SCALE;
        float gy = gy_raw * GYR_SCALE;
        float gz = gz_raw * GYR_SCALE;

        // Serializar como CSV compacto:  ax,ay,az,gx,gy,gz
        // Formato: 4 decimales para acelerómetro (resolución ~0.0001 g)
        //          2 decimales para giroscopio   (resolución ~0.01 °/s)
        snprintf(msgBuf, sizeof(msgBuf),
                 "%.4f,%.4f,%.4f,%.2f,%.2f,%.2f",
                 ax, ay, az, gx, gy, gz);

        // Enviar solo si hay conexión activa; si no, los datos se descartan
        // (evita acumulación de cola que desincronizaría la captura)
        if (wsConnected) {
            ws.sendTXT(msgBuf);
            sampleCount++;
        }
    }

    // Estadísticas en Serial cada 5 segundos
    uint32_t nowMs = millis();
    if (nowMs - lastStatsMs >= 5000) {
        float elapsed = (nowMs - lastStatsMs) / 1000.0f;
        float actualHz = sampleCount / elapsed;
        Serial.printf("[STATS] Muestras enviadas: %u | Frecuencia real: %.1f Hz | WiFi RSSI: %d dBm\n",
                      sampleCount, actualHz, WiFi.RSSI());
        sampleCount  = 0;
        lastStatsMs  = nowMs;
    }
}

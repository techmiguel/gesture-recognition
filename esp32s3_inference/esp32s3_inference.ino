#include <Wire.h>
#include <MPU6050.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "esp_task_wdt.h"

#include "gesture_model.h"

SET_LOOP_TASK_STACK_SIZE(32 * 1024);

static const int WINDOW_SIZE  = 100;
static const int NUM_AXES     = 6;
static const int NUM_CLASSES  = 5;
static const int TENSOR_ARENA_SIZE = 300 * 1024;

// Parámetros de normalización Z-score (de norm_params.npz)
static const float NORM_MEAN[NUM_AXES] = {  0.646844f,   0.202382f,  0.910041f,
                                           -10.908035f,  -9.341209f,  0.322231f };
static const float NORM_STD[NUM_AXES]  = {  0.296141f,   0.310030f,  0.035352f,
                                            57.859291f,  31.249041f, 26.466639f };

// Solo punteros como globals — cero constructores TFLite antes de setup()
static uint8_t*                tensor_arena = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

static const char* GESTURE_NAMES[NUM_CLASSES] = {
    "Abajo", "Arriba", "Derecha", "Izquierda", "Reposo"
};

MPU6050 mpu;
float window_buf[WINDOW_SIZE][NUM_AXES];
int   window_idx = 0;

void setup() {
    esp_task_wdt_deinit();

    Serial.begin(115200);
    delay(2000);
    Serial.println("=== BOOT ===");

    tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        Serial.println("ERROR: ps_malloc falló");
        while (true) { delay(1000); }
    }

    Wire.begin(8, 9);
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("ERROR: MPU6050 no responde");
        while (true) { delay(1000); }
    }
    Serial.println("MPU6050 OK");

    // Todos los objetos TFLite en heap, dentro de setup()
    auto* error_reporter = new tflite::MicroErrorReporter();
    auto* op_resolver    = new tflite::MicroMutableOpResolver<9>();
    op_resolver->AddConv2D();
    op_resolver->AddMaxPool2D();
    op_resolver->AddFullyConnected();
    op_resolver->AddSoftmax();
    op_resolver->AddReshape();
    op_resolver->AddExpandDims();
    op_resolver->AddShape();
    op_resolver->AddStridedSlice();
    op_resolver->AddPack();

    const tflite::Model* model = tflite::GetModel(gesture_model_int8_tflite);

    interpreter = new tflite::MicroInterpreter(
        model, *op_resolver, tensor_arena, TENSOR_ARENA_SIZE,
        error_reporter, nullptr, nullptr
    );

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors falló");
        while (true) { delay(1000); }
    }
    Serial.println("Modelo cargado OK");
}

void loop() {
    int16_t ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw;
    mpu.getMotion6(&ax_raw, &ay_raw, &az_raw, &gx_raw, &gy_raw, &gz_raw);

    float sample_scale_accel = 1.0f / 16384.0f;
    float sample_scale_gyro  = 1.0f / 131.0f;

    window_buf[window_idx][0] = ax_raw * sample_scale_accel;
    window_buf[window_idx][1] = ay_raw * sample_scale_accel;
    window_buf[window_idx][2] = az_raw * sample_scale_accel;
    window_buf[window_idx][3] = gx_raw * sample_scale_gyro;
    window_buf[window_idx][4] = gy_raw * sample_scale_gyro;
    window_buf[window_idx][5] = gz_raw * sample_scale_gyro;

    window_idx = (window_idx + 1) % WINDOW_SIZE;

    if (window_idx == 0) {
        TfLiteTensor* input       = interpreter->input(0);
        float         input_scale = input->params.scale;
        int           input_zp    = input->params.zero_point;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            for (int j = 0; j < NUM_AXES; j++) {
                float val = (window_buf[i][j] - NORM_MEAN[j]) / NORM_STD[j];
                int32_t q32 = (int32_t)roundf(val / input_scale) + input_zp;
                input->data.int8[i * NUM_AXES + j] =
                    (int8_t)(q32 < -128 ? -128 : (q32 > 127 ? 127 : q32));
            }
        }

        if (interpreter->Invoke() == kTfLiteOk) {
            TfLiteTensor* output   = interpreter->output(0);
            float         out_scale = output->params.scale;
            int           out_zp   = output->params.zero_point;

            float max_score      = -1.0f;
            int   predicted_class = -1;
            for (int i = 0; i < NUM_CLASSES; i++) {
                float score = (output->data.int8[i] - out_zp) * out_scale;
                if (score > max_score) {
                    max_score       = score;
                    predicted_class = i;
                }
            }
            const float CONFIDENCE_THRESHOLD = 0.90f;
            if (max_score >= CONFIDENCE_THRESHOLD) {
                Serial.print("Gesto: ");
                Serial.print(GESTURE_NAMES[predicted_class]);
                Serial.print(" | Confianza: ");
                Serial.println(max_score);
            }
        }
    }
    delay(10);
}

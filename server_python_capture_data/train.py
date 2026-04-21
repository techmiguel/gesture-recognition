# =============================================================================
# Pipeline de Entrenamiento — Reconocimiento de Gestos con MPU6050
#
# Clases: arriba, abajo, izquierda, derecha, reposo
# Entrada: 5 archivos CSV (uno por etiqueta) en el mismo directorio
# Salida:
#   - gesture_model.keras           (modelo Keras completo)
#   - gesture_model_float32.tflite  (modelo float32 para EloquentTinyML)
#   - gesture_model.h               (array C para incluir en firmware)
#   - norm_params.npz               (media y std para hardcodear en firmware)
#
# Dependencias:
#   pip install tensorflow scikit-learn numpy pandas matplotlib
# =============================================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# ── Configuración ─────────────────────────────────────────────────────────────
WINDOW_SIZE  = 100
NUM_AXES     = 6
NUM_CLASSES  = 5
BATCH_SIZE   = 32
EPOCHS       = 100
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15

LABELS = ["arriba", "abajo", "izquierda", "derecha", "reposo"]

CSV_FILES = {label: f"data_{label}.csv" for label in LABELS}

# ── 1. Carga de datos ─────────────────────────────────────────────────────────
print("=" * 60)
print("  FASE 1 — Carga de datos")
print("=" * 60)

frames = []
for label, filepath in CSV_FILES.items():
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"No se encontró '{filepath}'. "
            f"Asegúrate de ejecutar este script en la misma carpeta que los CSV."
        )
    df = pd.read_csv(filepath)
    df["label"] = label
    frames.append(df)
    print(f"  {label:<12} : {len(df):>4} muestras  ({filepath})")

df_all = pd.concat(frames, ignore_index=True)
print(f"\n  Total: {len(df_all)} muestras | {df_all['label'].value_counts().to_dict()}")

counts = df_all["label"].value_counts()
min_count = counts.min()
max_count = counts.max()
if max_count / min_count > 1.5:
    print(f"\n  [WARN] Desbalance detectado (ratio {max_count/min_count:.2f}x). "
          f"Considera recolectar más muestras de: {counts.idxmin()}")

# ── 2. Preprocesamiento ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 2 — Preprocesamiento")
print("=" * 60)

feature_cols = [c for c in df_all.columns if c != "label"]
X_raw = df_all[feature_cols].values.astype(np.float32)
y_raw = df_all["label"].values

X = X_raw.reshape(-1, WINDOW_SIZE, NUM_AXES)

le = LabelEncoder()
le.fit(LABELS)
y  = le.transform(y_raw)
print(f"  Clases codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=42
)

mean = X_train.mean(axis=(0, 1), keepdims=True)
std  = X_train.std(axis=(0, 1), keepdims=True)

X_train_norm = (X_train - mean) / (std + 1e-8)
X_test_norm  = (X_test  - mean) / (std + 1e-8)

np.savez("norm_params.npz", mean=mean, std=std)
print(f"\n  Parámetros de normalización guardados en norm_params.npz")
print(f"  mean (por eje): {mean.flatten().tolist()}")
print(f"  std  (por eje): {std.flatten().tolist()}")
print(f"\n  Train: {X_train_norm.shape} | Test: {X_test_norm.shape}")

# ── 3. Definición del modelo ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 3 — Modelo")
print("=" * 60)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW_SIZE, NUM_AXES)),

    tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
], name="gesture_classifier")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── 4. Entrenamiento ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 4 — Entrenamiento")
print("=" * 60)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=6, factor=0.5, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train_norm, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ── 5. Evaluación ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 5 — Evaluación")
print("=" * 60)

loss, acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\n  Test loss    : {loss:.4f}")
print(f"  Test accuracy: {acc:.4f}  ({acc*100:.1f}%)")

y_pred = np.argmax(model.predict(X_test_norm, verbose=0), axis=1)
print("\n  Reporte por clase:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="train acc")
axes[0].plot(history.history["val_accuracy"], label="val acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Época")
axes[0].legend()

im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
axes[1].set_title("Matriz de Confusión")
axes[1].set_xticks(range(NUM_CLASSES))
axes[1].set_yticks(range(NUM_CLASSES))
axes[1].set_xticklabels(le.classes_, rotation=45, ha="right")
axes[1].set_yticklabels(le.classes_)
axes[1].set_ylabel("Real")
axes[1].set_xlabel("Predicho")
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    axes[1].text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\n  Gráficas guardadas en training_results.png")

# ── 6. Guardar modelo Keras ───────────────────────────────────────────────────
model.save("gesture_model.keras")
print("\n  Modelo Keras guardado: gesture_model.keras")

# ── 7. Conversión a TFLite float32 ───────────────────────────────────────────
# Se usa float32 puro (sin cuantización) porque EloquentTinyML predict(float*)
# escribe directamente en in->data.f[] — requiere tensor float32, no int8.
print("\n" + "=" * 60)
print("  FASE 6 — Conversión TFLite float32")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Sin cuantización — float32 nativo
tflite_model = converter.convert()

with open("gesture_model_float32.tflite", "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"  Modelo float32 guardado: gesture_model_float32.tflite ({size_kb:.1f} KB)")

# ── 8. Generar array C para firmware ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  FASE 7 — Generación de gesture_model.h")
print("=" * 60)

hex_array = ", ".join(f"0x{b:02x}" for b in tflite_model)
var_name  = "gesture_model_int8_tflite"   # nombre mantenido para compatibilidad con firmware

header_content = f"""\
// Auto-generado por train.py — no editar manualmente
// Modelo: gesture_model_float32.tflite
// Tamaño: {size_kb:.1f} KB  ({len(tflite_model)} bytes)
// NOTA: modelo float32, compatible con EloquentTinyML predict(float*)

#ifndef GESTURE_MODEL_H
#define GESTURE_MODEL_H

#include <stdint.h>

alignas(8) const uint8_t {var_name}[] = {{
  {hex_array}
}};

const unsigned int {var_name}_len = {len(tflite_model)};

#endif // GESTURE_MODEL_H
"""

with open("gesture_model.h", "w") as f:
    f.write(header_content)

print(f"  Array C guardado: gesture_model.h")

# ── 9. Resumen final ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN FINAL")
print("=" * 60)
print(f"  Test accuracy           : {acc*100:.1f}%")
print(f"  Tamaño modelo float32   : {size_kb:.1f} KB")
print(f"  Archivos generados:")
print(f"    gesture_model.keras")
print(f"    gesture_model_float32.tflite")
print(f"    gesture_model.h           <- copiar al directorio del firmware")
print(f"    norm_params.npz           <- valores para hardcodear en firmware")
print(f"    training_results.png")
print()
print("  Parámetros de normalización para el firmware:")
mean_flat = mean.flatten()
std_flat  = std.flatten()
axes_names = ["ax", "ay", "az", "gx", "gy", "gz"]
for i, name in enumerate(axes_names):
    print(f"    {name}: mean={mean_flat[i]:.6f}  std={std_flat[i]:.6f}")
print()
print("  IMPORTANTE: Actualizar TENSOR_ARENA_SIZE en el firmware a 300 * 1024")
print("=" * 60)
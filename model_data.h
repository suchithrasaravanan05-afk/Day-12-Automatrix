import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Dataset Generation
np.random.seed(42)
X = np.random.uniform(0, 4095, size=(2000, 1)).astype(np.float32)
y = np.zeros(len(X), dtype=np.int32)
y[(X[:, 0] > 1200) & (X[:, 0] <= 2800)] = 1  # Normal
y[X[:, 0] > 2800] = 2                         # Bright
X_norm = X / 4095.0

# 2. Build and Train
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_norm, y, epochs=45, batch_size=16, validation_split=0.2, verbose=0)

# 3. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# 4. Generate C Header for ESP32
hex_array = ", ".join([f"0x{b:02x}" for b in tflite_model])
with open("model_data.h", "w") as f:
    f.write(f"#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\nconst unsigned char g_model[] = {{ {hex_array} }};\nconst int g_model_len = {len(tflite_model)};\n\n#endif\n")
print("Saved model_data.h successfully!")

# 5. Visualise TFLite Inference Output as a Graph
sweep_adc = np.linspace(0, 4095, 300, dtype=np.float32).reshape(-1, 1)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

probs = []
for val in sweep_adc:
    interpreter.set_tensor(input_idx, (val / 4095.0).reshape(1, 1))
    interpreter.invoke()
    probs.append(interpreter.get_tensor(output_idx)[0])
probs = np.array(probs)

plt.figure(figsize=(10, 4))
plt.plot(sweep_adc, probs[:, 0], label='Dark Prob', color='purple')
plt.plot(sweep_adc, probs[:, 1], label='Normal Prob', color='green')
plt.plot(sweep_adc, probs[:, 2], label='Bright Prob', color='orange')
plt.axvline(1200, color='grey', linestyle='--', label='Threshold 1 (1200)')
plt.axvline(2800, color='grey', linestyle=':', label='Threshold 2 (2800)')
plt.title("TFLite Model Output Probabilities across ADC Range")
plt.xlabel("Raw ADC Reading (0 - 4095)")
plt.ylabel("Predicted Probability")
plt.legend()
plt.grid(True)
plt.show()

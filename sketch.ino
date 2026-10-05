#include <EloquentTinyML.h>
#include "model_data.h"

#define NUMBER_OF_INPUTS 1
#define NUMBER_OF_OUTPUTS 3
#define TENSOR_ARENA_SIZE 2 * 1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;

const int SENSOR_PIN = 34;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  if (!ml.begin(g_model)) {
    Serial.println("Model init failed!");
    while (1);
  }
}

void loop() {
  int raw_val = analogRead(SENSOR_PIN);
  float input[1] = { raw_val / 4095.0f };
  float output[3];

  // Run TFLite inference
  ml.predict(input, output);

  // Print in Serial Plotter format for a real-time graph
  Serial.print("RawADC_Div1000:");
  Serial.print(raw_val / 1000.0f);
  Serial.print(" Dark:");
  Serial.print(output[0]);
  Serial.print(" Normal:");
  Serial.print(output[1]);
  Serial.print(" Bright:");
  Serial.println(output[2]);

  delay(100);
}


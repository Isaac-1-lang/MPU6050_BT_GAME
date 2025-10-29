#include <Wire.h>
#include <MPU6050_tockn.h>
#include <SoftwareSerial.h>

// --- MPU6050 setup ---
MPU6050 mpu6050(Wire);

// --- Bluetooth setup ---
// Connect HC-05: TX → 11, RX → 10
SoftwareSerial bt(10, 11);  // RX, TX

void setup() {
  Serial.begin(9600);   // For serial monitor
  bt.begin(38400);      // For Bluetooth module

  Serial.println("Initializing MPU6050...");
  Wire.begin();
  mpu6050.begin();
  mpu6050.calcGyroOffsets();  // Calibrate while sensor is still

  Serial.println("MPU6050 Ready!");
  bt.println("MPU6050 Bluetooth Transmission Started!");
}

void loop() {
  mpu6050.update();  // Read new data from sensor

  // --- Collect sensor data ---
  float accX = mpu6050.getAccX();
  float accY = mpu6050.getAccY();
  float accZ = mpu6050.getAccZ();

  float gyroX = mpu6050.getGyroX();
  float gyroY = mpu6050.getGyroY();
  float gyroZ = mpu6050.getGyroZ();

  // --- Create a formatted string of data ---
  String data = "Accel => X:" + String(accX, 2) + " Y:" + String(accY, 2) + " Z:" + String(accZ, 2);
  data += " | Gyro => X:" + String(gyroX, 2) + " Y:" + String(gyroY, 2) + " Z:" + String(gyroZ, 2);

  // --- Send data to Serial Monitor and Bluetooth ---
  Serial.println(data);
  bt.println(data);

  delay(100);
}
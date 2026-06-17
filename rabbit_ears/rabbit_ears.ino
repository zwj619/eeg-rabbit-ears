/***************************************************
  兔耳朵 — PCA9685 双舵机 EEG 脑控
  睁眼: 耳朵放下,  闭眼: 耳朵竖起
 ****************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 150
#define SERVOMAX 400
#define SERVO_FREQ 50

int value;
int lastValue = '1';  // 初始默认竖起

void setup() {
    Serial.begin(115200);

    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(SERVO_FREQ);

    delay(10);
    // 启动时竖起
    for (uint16_t i = 0; SERVOMIN + i <= SERVOMAX; i++) {
        pwm.setPWM(0, 0, SERVOMIN + i);
        pwm.setPWM(4, 0, SERVOMAX - i);
        delay(2);
    }
}

void loop() {
    if (Serial.available() > 0) {
        value = Serial.read();

        // 竖起 (闭眼)
        if (value == '1' && lastValue != '1') {
            for (uint16_t i = 0; SERVOMIN + i <= SERVOMAX; i++) {
                pwm.setPWM(0, 0, SERVOMIN + i);
                pwm.setPWM(4, 0, SERVOMAX - i);
                delay(2);
            }
            lastValue = '1';
        }
        // 放下 (睁眼)
        else if (value == '0' && lastValue != '0') {
            for (uint16_t i = 0; SERVOMIN + i <= SERVOMAX; i++) {
                pwm.setPWM(0, 0, SERVOMAX - i);
                pwm.setPWM(4, 0, SERVOMIN + i);
                delay(2);
            }
            lastValue = '0';
        }
    }
    delay(1);
}

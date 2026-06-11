// PS5 BT MAC: 7C:66:EF:44:90:C9

#include <Arduino.h>
#include <BTS7960.h>
#include "ps5Controller.h"

// ===== MOTOR DRIVER PINS =====
#define LEFT_L_EN   25
#define LEFT_R_EN   26
#define LEFT_L_PWM  32
#define LEFT_R_PWM  33

#define RIGHT_L_EN   4
#define RIGHT_R_EN   18
#define RIGHT_L_PWM  19
#define RIGHT_R_PWM  5

// ===== UART2 → RDK X5 =====
#define RDK_RX 16
#define RDK_TX 17

// ===== TUNING =====
const int DEAD      = 10;   // trigger deadband
const int MIN_SPEED = 80;   // minimum PWM to overcome stiction
const int MAX_SPEED = 200;  // PWM cap (≤255)
const int TURN_DIFF = 53;   // speed difference between fast/slow side when turning
const int LOOP_MS   = 20;   // main loop delay

static const char* PS5_MAC = "7c:66:ef:44:90:c9";

BTS7960 leftSide (LEFT_L_EN,  LEFT_R_EN,  LEFT_L_PWM,  LEFT_R_PWM);
BTS7960 rightSide(RIGHT_L_EN, RIGHT_R_EN, RIGHT_L_PWM, RIGHT_R_PWM);

// ─────────────────────────────────────────────────────────────────────────────

void stopAll() {
  leftSide.Stop();
  rightSide.Stop();
}

// speed > 0 → forward, speed < 0 → reverse
// analog_x_val: left stick X (-128..127), negative = left, positive = right
void setSpeed(int speed, int analog_x_val) {
  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);

  if (speed == 0) {
    stopAll();
    return;
  }

  int mag = abs(speed);
  if (mag < MIN_SPEED) mag = MIN_SPEED;
  if (mag > MAX_SPEED) mag = MAX_SPEED;

  int mag_fast = min(mag + TURN_DIFF, MAX_SPEED);

  if (speed > 0) {
    // ── Forward ──────────────────────────────────────────────────────────────
    if (analog_x_val < -50) {
      // turn left: right side faster
      rightSide.TurnLeft((uint8_t)mag_fast);
      leftSide.TurnRight((uint8_t)mag);
    } else if (analog_x_val > 50) {
      // turn right: left side faster
      rightSide.TurnLeft((uint8_t)mag);
      leftSide.TurnRight((uint8_t)mag_fast);
    } else {
      // straight
      rightSide.TurnLeft((uint8_t)mag);
      leftSide.TurnRight((uint8_t)mag);
    }
  } else {
    // ── Reverse ───────────────────────────────────────────────────────────────
    if (analog_x_val < -50) {
      rightSide.TurnRight((uint8_t)mag_fast);
      leftSide.TurnLeft((uint8_t)mag);
    } else if (analog_x_val > 50) {
      rightSide.TurnRight((uint8_t)mag);
      leftSide.TurnLeft((uint8_t)mag_fast);
    } else {
      rightSide.TurnRight((uint8_t)mag);
      leftSide.TurnLeft((uint8_t)mag);
    }
  }
}

void onSpotTurn(int analog_x_val) {
  if (analog_x_val < -50) {
    // spin left: right forward, left backward
    rightSide.TurnLeft((uint8_t)120);
    leftSide.TurnLeft((uint8_t)120);
  } else {
    // spin right: left forward, right backward
    rightSide.TurnRight((uint8_t)120);
    leftSide.TurnRight((uint8_t)120);
  }
}

// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RDK_RX, RDK_TX);

  leftSide.Enable();
  rightSide.Enable();
  stopAll();
  delay(300);

  ps5.begin(PS5_MAC);
  Serial.println("Connecting to PS5 controller...");
  while (!ps5.isConnected()) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nConnected!");
}

void loop() {
  if (!ps5.isConnected()) {
    stopAll();
    delay(200);
    return;
  }

  // ── Read inputs ────────────────────────────────────────────────────────────
  int  r2           = ps5.R2Value();
  int  l2           = ps5.L2Value();
  int  left_analog_x = ps5.LStickX();

  bool x_pressed    = ps5.Cross();
  bool sq_pressed   = ps5.Square();
  bool tr_pressed   = ps5.Triangle();
  bool o_pressed    = ps5.Circle();
  bool dp_up        = ps5.Up();
  bool dp_down      = ps5.Down();
  bool dp_left      = ps5.Left();
  bool dp_right     = ps5.Right();

  // ── Pose commands (face buttons + d-pad) ──────────────────────────────────
  // Each sends a string over UART2 to the RDK X5 arm commander node
  if (x_pressed)  { Serial2.println("pose_1"); delay(300); return; }
  if (sq_pressed) { Serial2.println("pose_2"); delay(300); return; }
  if (tr_pressed) { Serial2.println("pose_3"); delay(300); return; }
  if (o_pressed)  { Serial2.println("pose_4"); delay(300); return; }
  if (dp_down)    { Serial2.println("pose_5"); delay(300); return; }
  if (dp_left)    { Serial2.println("pose_6"); delay(300); return; }
  if (dp_up)      { Serial2.println("pose_7"); delay(300); return; }
  if (dp_right)   { Serial2.println("pose_8"); delay(300); return; }

  // ── Deadband ───────────────────────────────────────────────────────────────
  if (r2 < DEAD) r2 = 0;
  if (l2 < DEAD) l2 = 0;

  // ── Drive logic ───────────────────────────────────────────────────────────
  // On-spot turn: no triggers, stick pushed left or right
  if (r2 == 0 && l2 == 0 && (left_analog_x < -50 || left_analog_x > 50)) {
    onSpotTurn(left_analog_x);
    delay(LOOP_MS);
    return;
  }

  int cmd = 0;
  if      (r2 > 0 && l2 == 0) cmd =  r2;   // forward
  else if (l2 > 0 && r2 == 0) cmd = -l2;   // reverse
  // both pressed or neither → cmd stays 0 → stopAll()

  setSpeed(cmd, left_analog_x);

  delay(LOOP_MS);
}
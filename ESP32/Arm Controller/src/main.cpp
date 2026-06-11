#include <ESP32Servo.h>

#define base_pin     12
#define shoulder_pin 14
#define elbow_pin    19
#define wrist1_pin   16
#define wrist2_pin   17
#define gripper_pin  33

Servo base, shoulder, elbow, wrist1, wrist2, gripper;
Servo* servos[6];

int current_pos[6] = {90, 90, 45, 90, 90, 90};
int target_pos[6]  = {90, 90, 90, 90, 90, 90};

// Minimum step delay per joint (full speed, cruise phase)
const int min_delay[6] = {
    12,  // base
    20,  // shoulder — highest load
    18,  // elbow
    10,  // wrist1
    10,  // wrist2
    8    // gripper
};

// Maximum step delay per joint (start and end of movement)
const int max_delay[6] = {
    40,  // base
    70,  // shoulder
    60,  // elbow
    35,  // wrist1
    35,  // wrist2
    25   // gripper
};

// Ramp steps per joint (acceleration/deceleration window)
const int ramp_steps[6] = {
    15,  // base
    20,  // shoulder
    18,  // elbow
    12,  // wrist1
    12,  // wrist2
    10   // gripper
};

int steps_taken[6] = {0, 0, 0, 0, 0, 0};

// Serial parsing
uint8_t idx       = 0;
uint8_t value_idx = 0;
char value[4]     = "000";

// Trapezoidal velocity profile
int compute_delay(int joint){
    int distance = abs(target_pos[joint] - current_pos[joint]);
    int steps    = steps_taken[joint];
    int ramp     = ramp_steps[joint];
    int min_d    = min_delay[joint];
    int max_d    = max_delay[joint];

    if(steps < ramp){
        return max_d - ((max_d - min_d) * steps / ramp);
    }
    if(distance < ramp){
        return min_d + ((max_d - min_d) * (ramp - distance) / ramp);
    }
    return min_d;
}

void setup(){
    servos[0] = &base;
    servos[1] = &shoulder;
    servos[2] = &elbow;
    servos[3] = &wrist1;
    servos[4] = &wrist2;
    servos[5] = &gripper;

    base.attach(base_pin);
    shoulder.attach(shoulder_pin);
    elbow.attach(elbow_pin);
    wrist1.attach(wrist1_pin);
    wrist2.attach(wrist2_pin);
    gripper.attach(gripper_pin);

    for(int i = 0; i < 6; i++){
        servos[i]->write(current_pos[i]);
    }

    delay(1000);
    Serial.begin(115200);
    Serial.setTimeout(1);
}

void loop(){
    // 1. Read serial — always first, never blocked
    // Protocol: <joint_char><3-digit-angle>,
    // e.g. "b090,s045,e120,w060,x090,g030,"
    // b = base, s = shoulder, e = elbow
    // w = wrist1, x = wrist2, g = gripper
    while(Serial.available()){
        char ch = Serial.read();
        Serial.print(ch);
        if     (ch == 'b') { idx = 0; value_idx = 0; }
        else if(ch == 's') { idx = 1; value_idx = 0; }
        else if(ch == 'e') { idx = 2; value_idx = 0; }
        else if(ch == 'w') { idx = 3; value_idx = 0; }
        else if(ch == 'x') { idx = 4; value_idx = 0; }
        else if(ch == 'g') { idx = 5; value_idx = 0; }
        else if(ch == ','){
            int new_target = atoi(value);

            if(new_target != target_pos[idx]){
                target_pos[idx] = new_target;
                steps_taken[idx] = 0;
            }

            value[0] = '0'; value[1] = '0';
            value[2] = '0'; value[3] = '\0';
            value_idx = 0;
        }
        else{
            if(value_idx < 3){
                value[value_idx++] = ch;
            }
        }
    }

    // 2. Step each servo with velocity profiling
    static unsigned long last_step[6] = {0, 0, 0, 0, 0, 0};
    unsigned long now = millis();

    for(int i = 0; i < 6; i++){
        if(current_pos[i] != target_pos[i]){
            int delay_ms = compute_delay(i);

            if(now - last_step[i] >= (unsigned long)delay_ms){
                last_step[i] = now;

                if(current_pos[i] < target_pos[i]) current_pos[i]++;
                else                                current_pos[i]--;

                steps_taken[i]++;
                servos[i]->write(current_pos[i]);
            }
        }
        else{
            steps_taken[i] = 0;
        }
    }
}

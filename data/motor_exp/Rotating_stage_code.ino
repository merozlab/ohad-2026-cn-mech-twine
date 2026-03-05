
// Rotation stage

// This piece of code allows to control the "rotation stage" based on a simple DC motor and a 3D printed gear system.
// the worm gear is 1:60 and the 2 other gears have a 1:3 gear ratio, giving a speed reduction of a factor of 180. 
// The ratio is equal to the number of gear teeth divided by the number of starts/threads on the worm.

// You may want to adapt the variables according to your experiment.

// History:
// 2023.09.04: Add option to start/stop motor after given duration
// 2023.08.10: Creation

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// include libraries for later use
#include <SD.h>
//#include <DHT.h>

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// Declare PINS, constants and variables

// initialize digital pins as an output for the motor
const int motorin1 =  3;      // the number of the motor pin
const int motorin2 =  2;      // the number of the motor pin  
const int motorenable =  9;      // the number of the motor pin

const int motor_speed1 = 180; // speed on a scale from 0 to 255
const unsigned long power_duration1 = 0; // 1000UL*60*60*7.5; // time to run motor: 1000 for seceonds * 60 for minutes * 60 for hours * x minutes

const int motor_speed2 = 110; // speed on a scale from 0 to 255
const unsigned long power_duration2 = 1000UL*60*60*7.5; // time to run motor: 1000 for seceonds * 60 for minutes * 60 for hours * x minutes

// Time variables:
unsigned long start_time = millis();  // start time

void setup() {
  Serial.begin(9600);

  pinMode(motorin1, OUTPUT);
  pinMode(motorin2, OUTPUT);
  pinMode(motorenable, OUTPUT);

  digitalWrite(motorin1, HIGH);
  digitalWrite(motorin2, LOW); 
  analogWrite(motorenable, 255); 
}

// the loop function runs until the if is false

void loop() {
  // for first power_duration1 minutes- run direction 1. 
  if (millis() - start_time<= power_duration1) { // 
    // start turning

    digitalWrite(motorin1, HIGH);
    digitalWrite(motorin2, LOW); 
    analogWrite(motorenable, motor_speed1); // wont turn under pwm of 100 ?
  }
  
  // for power_duration2 minutes- run direction 2 at speed 2
  else if (millis() - start_time<= power_duration1+power_duration2) { // 
    // start turning

    digitalWrite(motorin1, LOW);
    digitalWrite(motorin2, HIGH); 
    analogWrite(motorenable, motor_speed2); // wont turn under pwm of 100 ?
  }
  else {
    digitalWrite(motorin1, LOW);
    digitalWrite(motorin2, LOW);   
  }
    // try to give the motor a start at pwm: 255 and then reduce to lower pwm values for slower rotation rate.loopcount = loopcount+1;
}

//digitalWrite(motorin1, LOW); 
//digitalWrite(motorin2, LOW); 

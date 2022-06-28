#include "max6675.h"
#include "Adafruit_VL53L0X.h"
// https://learn.adafruit.com/thermocouple/
Adafruit_VL53L0X lox = Adafruit_VL53L0X();

int thermoDO = 4;
int thermoCS = 5;
int thermoCLK = 6;

MAX6675 thermocouple(thermoCLK, thermoCS, thermoDO);
int vccPin = 3;
int gndPin = 2;
void setup() {
  Serial.begin(9600);
  
  // wait until serial port opens for native USB devices
  // wait until serial port opens for native USB devices
  while (! Serial) {
    delay(1);
  }
  
  Serial.println("MAX6675 and Adafruit VL53L0X test");
  pinMode(vccPin, OUTPUT); digitalWrite(vccPin,HIGH);
  pinMode(gndPin, OUTPUT); digitalWrite(gndPin,LOW);

  // wait for MAX chip to stabilize
  delay(500);
 
  if (!lox.begin()) {
    Serial.println(F("Failed to boot VL53L0X"));
    while(1);
  }
}

void loop() {
  
  // Start a new line
  Serial.print("\n");
  // TC readout, print current temp in deg F
  Serial.print(thermocouple.readFahrenheit()); 
  int i;
    // Lidar readout, print twice because TC requires 150 longer to reload
   for(i=0;i<3;i++){//make sure to update delay with the number of loops
    VL53L0X_RangingMeasurementData_t measure;
    lox.rangingTest(&measure, false);
    if (measure.RangeStatus != 4) {  // phase failures have incorrect data
      Serial.print(measure.RangeMilliMeter);
      Serial.print(" ");
      delay(84);
    } else {
      Serial.print("000");
    }
   }
   // For the MAX6675 to update, you must delay AT LEAST 250ms between reads!
   // VL53L0X only needs 100ms so the 250ms wins. 84+84+84>250ms
   //delay(83);
}


//Serial.print("C = "); 
   //Serial.println(thermocouple.readCelsius());

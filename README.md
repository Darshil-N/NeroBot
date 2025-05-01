NeroBot 
Software part of NeroBot 

Technology used 

1. For Designing and CAD tools
  a. For 3D structural design
    --> Software used :- Fusion 360
    --> This will be used to Design the 3D structure of the structure of the jellyfish and part designing as well

2. For simulation
   a. For motion simulation of robot
   --> Software used :- WeBots
   --> This Software will be used to simulate the motion of the bot and to find any mechanical obstruction within the bot itself during the motion and make it ready for real world

    b. For Real world Physics
   --> Software used :- Unity Engine
   --> For real world Physics and condition we will use the Unity Game engine where we will train our bot the basic motion and prepare for real world condition without damaging the bot

   c. For Animation and Visualization
   --> Software Used :- Blender
   --> Blender will be uesd for basic visualisation of the bot moving and record it in different condition and able to record the movement for any required changes in the model

3. Embedded System
      a.  Low-Level Embedded Software (For ESP32)
      --> Language used :- C/C++
      --> Frameworks :- Arduino Core For ESP32
      --> IDE :- Arduino IDE or PlatformIO

      --> Function :- Servo control
          Library :- Servo.h/ESP32Servo.h
          purpose :- Controls Tentacle movement via Pulse width modulation(PWM)

      --> Function :- Sensor Readings
          Library :- Wire.h/Adafruit_Sensors/Adafruit_MPU6050
          purpose :- An Inertial Measure Unit(IMU), Inter-Integrated Circuit(I2C)

      --> Function :- Depth/pressure measurement
          Library :- UART(Universal Asynchronous Receiver/Transmitter) Driver
          Purpose :- This part will manage the communication between microcontroller and Depth/pressure sensors

      --> Function :- Safety Protocol
          Library :- 	digitalRead(), safe_shutdown(), surface_protocol(), Watchdog timer
          Purpose :- This part of the code is used to detect leaks or if the bot gets damaged it will resurface itself to the nearest docker and will all the implement the emergency protocols

      --> Function :- Serial Communication
          Library :- HardwareSerial, Serial2
          Purpose :- This part of the ESP32 code will help communication between ESP32 and Rasberry pie for further processing and communication of the sensors with cloud

   b. High-Level Embedded Software (Raspberry Pi)
     --> Language Used :- Python 3.9+
     --> OS :- Raspberry Pi OS (Debian Lite)

     --> Function :- Inertial Measurement Unit (IMU) Handeling
         Libraries :- RTIMULib, smbus2, Adafruit_MPU9250
         Purpose :- This part of the code will be used for the advance internal meachanism control like its position, angle, force, pressure, etc.

     --> Function :- Sonar Sensors
         Libraries :- RPi.GPIO, HCSR04
         Purpose :- This part of the code will be used for predator detection and with real time of flight calculation using ultrasonic sonar sensor to avoid the predator

     --> Function :- Camera Vision
         Libraries :- OpenCV, cv2.VideoCapture()
         Purpose :- Captures the real time live camera feed aand fead it to the AI system which will decide the further path and movement and can give a surveillance on the bot using the cloud

     --> Function :- Leak / Emergency
         Libraries :- GPIO.add_event_detect()
         Purpose :- This part of the code enables bot to trigger a calback function when specific emergency is detected on that pin. This allows the automatic shift of code from AI to static emergency code        without any manual supervision

     --> Function :- Bot to docker communication
         Libraries :- pyLoRa, LoRa, UART
         Purpose :- This part of the code will send the data from bot to docker via LoRa(if shallow) or acoustic modem (for deep waters)

     --> Function :- Data compression
         Libraries :- ujson, json
         Purpose :- Using this protocols we will compress the sending data to minimize the transmission size and hence reducing the transmission time

     --> Function :- Safety Protocols
         Libraries :- GPIO + ISR (ESP32/RPi), Timer-based UART ping/ACK logic
         Purpose :- This will again use the Watchdog Protocol to ensure the safety of the bot where it will send the urgent signal for any interruption stopping rest of the messages and ensuring the constant connection between docker and bot

     --> Function :- Docker to Cloud Communication
         Libraries :- paho-mqtt, firebase-admin, pyserial, SQLite3, CSV, requests, urllib3, ssl, MQTTS/HTTPS, Flask, Node-RED, or Kivy, Queue / FIFO buffer
         Purpose :- Pushes real-time data from Docker to cloud securely, prioritizing urgent messages and confirming packet receipt with fail-safe local storage. Manages connectivity  and supports cloud-pulled AI/control updates. Allows offline diagnostics and manual commands during field ops.

4. Artificial and Machine learning

  a.  AI Architecture Overview
   --> Function :- Movement Control AI
       Libraries :- SAC (Soft Actor-Critic) via Stable-Baselines3 + PyTorchh
       Purpose :- Reinforcement Learning (RL) agent which will be trained in Unity engine to optimize the propulsion through rhythmic tentacle pulsing.

  -->  Function :- Vision-Based Object Detection
       Libraries :- YOLOv8 (PyTorch), TorchScript.
       Purpose :- Convolutional neural networks(CNN) to detect marine life and decide the further path based on camera sensors in the bot which will run on Raspberry pi via Torchscript

  --> Function :- Inference Engine	
      Libraries :- TorchScript / ONNX
      Purpose :- Converts PyTorch models into optimized formats for inference on edge hardware (Raspberry Pi).

   -->  Function :- Decision Layer / Hybrid AI
        Libraries :- 	Hybrid: Rule-based + SAC Output
        Purpose :- this part of the AI will take the sensor inputs and using this inputs will take decisions such as avoid, dive, surface

b. AI Model Training
   --> Function :- SAC Training Loop
       Libraries :- Stable-Baselines3 + PyTorch
       Purpose :- Trains a soft actor-critic agent using replay buffers and stochastic policy sampling in Unity ML-Agents.

  --> Function :- Vision Training
      Libraries :- PyTorch + YOLOv5 Framework
      Purpose :-Object detector trained using labeled underwater datasets with data augmentation (blur, color shift, noise)

  --> Function :- Experiment Tracking
      Libraries :- Weights & Biases / TensorBoard
      Purpose :- this part of AI will be used to develop the neural network whivh will be used give rewards, monitor the condition bot, etc.

  --> Function :- Dataset Annotation
      Libraries :- CVAT / Roboflow
      Purpose :- Manual annotation of underwater debris for bounding box detection training.

  c. Vision pipeline  
    --> Function :- Frame Capture
      Libraries :- 	OpenCV (cv2.VideoCapture)
      Purpose :- Captures frames from camera in real time.

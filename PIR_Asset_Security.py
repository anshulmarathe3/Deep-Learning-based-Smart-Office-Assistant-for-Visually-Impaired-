import serial
import pyttsx3

# Initialize pyttsx3 engine for text-to-speech
engine = pyttsx3.init()

# Function to speak the alert message
def speak_alert():
    engine.say("Someone is near Asset")
    engine.runAndWait()

# Set up serial connection to Arduino (make sure to adjust the COM port)
# You can find the correct COM port in the Arduino IDE or Device Manager
arduino = serial.Serial('COM3', 9600)  # Adjust the COM port to match your system

while True:
    # Read the serial input from Arduino
    if arduino.in_waiting > 0:
        message = arduino.readline().decode('utf-8').strip()
        print(f"Received: {message}")
        
        # If motion is detected, speak the alert
        if message == "Motion Detected":
            speak_alert()
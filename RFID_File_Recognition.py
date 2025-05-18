import serial
import pyttsx3
import time

# Setup text-to-speech engine
engine = pyttsx3.init('sapi5')  # Force Windows SAPI5 voice engine
engine.setProperty('rate', 150)  # Set speaking rate

# Setup serial port
ser = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # Give time for connection to establish

# RFID tag to file mapping
rfid_to_file = {
    "54008A4E8818": "Annual Performance Appraisal Report",
    "54008BD37478": "Departmental Budget Allocation File",
    "54008C8A2E7C": "Confidential Government Orders"
}

print("Listening for RFID tags...")

while True:
    if ser.in_waiting > 0:
        rfid_data = ser.readline().decode('utf-8').strip()
        print(f"Received RFID: {rfid_data}")

        if rfid_data in rfid_to_file:
            file_name = rfid_to_file[rfid_data]
            message = f"{file_name} detected"
            print(message)

            # Speak the detected file
            engine.say(message)
            engine.runAndWait()  # Only this is needed!

            time.sleep(1)  # small delay to avoid repeat
        else:
            print("Unknown RFID tag.")

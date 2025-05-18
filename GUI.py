import tkinter as tk
import subprocess
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Speak startup message
def speak_startup():
    engine.say("Smart Office Assistant Activated")
    engine.runAndWait()

# Speak shutdown message and close app
def speak_and_exit():
    engine.say("Smart Office Assistant Deactivated")
    engine.runAndWait()
    root.destroy()

# Feature functions with announcements
def run_facial_recognition():
    global facial_process
    facial_process = subprocess.Popen(["python", "C:/Users/mehka/Desktop/Smart Office Assistant/Facial_Recognition/Facial_Recognition_MTCNN.py"])
    engine.say("Facial Recognition Activated")
    engine.runAndWait()

def close_facial_recognition(event=None):
    if 'facial_process' in globals():
        facial_process.terminate()
    engine.say("Facial Recognition Deactivated")
    engine.runAndWait()

def run_text_to_speech():
    global tts_process
    tts_process = subprocess.Popen(["python", "C:/Users/mehka/Desktop/Smart Office Assistant/Image_Text_to_Speech.py"])
    engine.say("Document Text-to-Speech Activated")
    engine.runAndWait()

def close_text_to_speech(event=None):
    if 'tts_process' in globals():
        tts_process.terminate()
    engine.say("Document Text-to-Speech Deactivated")
    engine.runAndWait()

def run_rfid_file_recognition():
    global rfid_process
    rfid_process = subprocess.Popen(["python", "C:/Users/mehka/Desktop/Smart Office Assistant/RFID_File_Recognition.py"])
    engine.say("File Recognition Activated")
    engine.runAndWait()

def close_rfid_file_recognition(event=None):
    if 'rfid_process' in globals():
        rfid_process.terminate()
    engine.say("File Recognition Deactivated")
    engine.runAndWait()

def run_asset_security_monitoring():
    global asset_process
    asset_process = subprocess.Popen(["python", "C:/Users/mehka/Desktop/Smart Office Assistant/PIR_Asset_Security.py"])
    engine.say("Asset Security Monitoring Activated")
    engine.runAndWait()

def close_asset_security_monitoring(event=None):
    if 'asset_process' in globals():
        asset_process.terminate()
    engine.say("Asset Security Monitoring Deactivated")
    engine.runAndWait()

# Main window
root = tk.Tk()
root.title("Smart Office Assistant for Visually Impaired")
root.attributes('-fullscreen', True)
root.configure(bg="#f0f0f0")

# Press ESC to exit fullscreen
def exit_fullscreen(event):
    root.attributes('-fullscreen', False)
root.bind("<Escape>", exit_fullscreen)

# Handle window close
root.protocol("WM_DELETE_WINDOW", speak_and_exit)

# Bind Ctrl+E to close the window
root.bind("<Control-e>", lambda event: speak_and_exit())

# Title
title_frame = tk.Frame(root, bg="#f0f0f0")
title_frame.pack(pady=40)

title_label = tk.Label(
    title_frame,
    text="🤖 SMART OFFICE ASSISTANT 🤖",
    font=("Segoe UI", 48, "bold"),
    fg="#2c3e50",  # Elegant dark blue-grey
    bg="#f0f0f0"
)
title_label.pack()

subtitle_label = tk.Label(
    title_frame,
    text="Empowering the Visually Impaired",
    font=("Segoe UI", 20, "italic"),
    fg="#7f8c8d",  # Soft grey for subtitle
    bg="#f0f0f0"
)
subtitle_label.pack(pady=(2, 0))

# Style for buttons
button_style = {
    "width": 40,
    "height": 3,
    "font": ("Arial", 18, "bold"),  # Bold font for text
    "bd": 3,  # Border width
    "relief": "solid",  # Solid border for clear shape
    "highlightthickness": 0,  # Removes highlight border
    "activebackground": "#4CAF50",  # Active color when clicked
    "activeforeground": "white",  # Text color when clicked
    "bg": "#4CAF50",  # Default background color
    "fg": "white"  # Text color
}

# Buttons with curvy edges (using 'relief' and 'bd' for styling)
btn1 = tk.Button(root, text="Facial Recognition", command=run_facial_recognition, **button_style)
btn1.config(bg="#ff6f61")  # Bright Coral
btn1.pack(pady=20)

btn2 = tk.Button(root, text="Document Text-to-Speech", command=run_text_to_speech, **button_style)
btn2.config(bg="#42a5f5")  # Bright Sky Blue
btn2.pack(pady=20)

btn3 = tk.Button(root, text="RFID File Recognition", command=run_rfid_file_recognition, **button_style)
btn3.config(bg="#f4c430")  # Vibrant Yellow
btn3.pack(pady=20)

btn4 = tk.Button(root, text="Asset Security Monitoring", command=run_asset_security_monitoring, **button_style)
btn4.config(bg="#0BDA51")  # Fresh Leaf Green
btn4.pack(pady=20)

# Speak on load (delayed so GUI loads first)
root.after(500, speak_startup)

# Key bindings for controlling features
root.bind("<f>", lambda event: run_facial_recognition())   # F to start facial recognition
root.bind("<Control-f>", lambda event: close_facial_recognition())  # Ctrl+F to close facial recognition

root.bind("<g>", lambda event: run_text_to_speech())       # G to start text-to-speech
root.bind("<Control-g>", lambda event: close_text_to_speech())    # Ctrl+G to close text-to-speech

root.bind("<h>", lambda event: run_rfid_file_recognition())  # H to start RFID recognition
root.bind("<Control-h>", lambda event: close_rfid_file_recognition())  # Ctrl+H to close RFID recognition

root.bind("<j>", lambda event: run_asset_security_monitoring())  # J to start asset security monitoring
root.bind("<Control-j>", lambda event: close_asset_security_monitoring())  # Ctrl+J to close asset security monitoring

# Start GUI
root.mainloop()

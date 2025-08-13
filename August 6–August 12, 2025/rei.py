import RPi.GPIO as GPIO
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from RPLCD.i2c import CharLCD
from pyfingerprint.pyfingerprint import PyFingerprint
import threading
from datetime import datetime
import logging
from picamera2 import Picamera2
import cv2
from datetime import datetime
import time
import requests
import os
import ssl
import logging
import threading


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define the GPIO pins for rows, columns, and button
ROW_PINS = [18, 25, 8, 7]
COL_PINS = [5, 27, 17, 4]

BUTTON_PIN = 20  # Replace with the GPIO pin for the button

# Define GPIO pin for the solenoid lock
LOCK_PIN = 6  # Replace with your solenoid lock GPIO pin

# Keypad layout
KEYPAD = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'OK']
]

# Set the correct password
PASSWORD = "1234"

def setup():
    # Set up GPIO mode
    GPIO.setmode(GPIO.BCM)
    
    # Set up rows as output and columns as input
    for row in ROW_PINS:
        GPIO.setup(row, GPIO.OUT)
        GPIO.output(row, GPIO.LOW)
    
    for col in COL_PINS:
        GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    # Set up the solenoid lock pin as output
    GPIO.setup(LOCK_PIN, GPIO.OUT)
    GPIO.output(LOCK_PIN, GPIO.LOW)  # Ensure the lock is initially closed

    # Set up the button pin as input
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def read_keypad():
    for row_index, row_pin in enumerate(ROW_PINS):
        # Set the current row HIGH
        GPIO.output(row_pin, GPIO.HIGH)
        for col_index, col_pin in enumerate(COL_PINS):
            if GPIO.input(col_pin) == GPIO.HIGH:
                # Debounce the key press
                time.sleep(0.02)  # Short delay
                if GPIO.input(col_pin) == GPIO.HIGH:
                    key = KEYPAD[row_index][col_index]
                    # Wait for the key to be released
                    while GPIO.input(col_pin) == GPIO.HIGH:
                        pass
                    return key
        # Set the current row LOW
        GPIO.output(row_pin, GPIO.LOW)
    return None
    return input("Simulate Key Press: ").strip()  # Simulate user input

def open_lock():
    """Opens the lock for a specific duration."""
    print("Unlocking door...")
    GPIO.output(LOCK_PIN, GPIO.HIGH)  # Activate solenoid
    time.sleep(5)  # Keep the lock open for 5 seconds
    GPIO.output(LOCK_PIN, GPIO.LOW)  # Deactivate solenoid
    print("Door locked again.")

def keypad_mode():
    """Handles the keypad password input logic."""
    print("Keypad mode activated. Press 'A' to begin password entry.")
    entered_password = ""  # Initialize the password buffer
    a_pressed = False  # Flag to check if "A" is pressed before accepting the password
    a_pressed_time = 0  # Variable to store the time when "A" is pressed

def update_lcd_display(line1, line2=""):
    """
    Update the LCD display with two lines of text.
    """
    lcd.clear()
    lcd.write_string(line1[:16])  # First line
    if line2:
        lcd.cursor_pos = (1, 0)  # Move to the second line
        lcd.write_string(line2[:16])  # Second line


# Initialize the LCD
lcd = CharLCD('PCF8574', 0x27)

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Pin Definitions
PINS = {
    'button1': 23,  # Record unlock command
    'button2': 24,  # Verify voice command
    'button3': 25,  # Stop listening
    'button4': 26,  # Fingerprint verification
    'button5': 16,  # Register fingerprint
    'solenoid': 6,
    'buzzer': 22,
    'button6' : 0
    }

# Setup GPIO pins
for pin in PINS.values():
    if pin in [PINS['solenoid'], PINS['buzzer']]:
        GPIO.setup(pin, GPIO.OUT)
    else:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize fingerprint sensor
try:
    f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        raise ValueError('The given fingerprint sensor password is incorrect!')
except Exception as e:
    logger.error(f'Fingerprint sensor initialization failed: {str(e)}')
    exit(1)

picam2 = Picamera2()
picam2.configure(picam2.preview_configuration)  # Set up camera
picam2.start()

def generate():
    while True:
        # Capture frame from PiCamera2
        frame = picam2.capture_array()
        # Convert the frame to JPEG format for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        # Yield frame as a byte stream for HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Function to save the registered voice command
def save_voice_command(command):
    """Save the registered voice command to a file."""
    try:
        with open("voice_command.txt", "w") as file:
            file.write(command)
        logger.info("Voice command saved successfully.")
    except Exception as e:
        logger.error(f"Error saving voice command: {str(e)}")

# Function to load the saved voice command
def load_voice_command():
    """Load the registered voice command from a file."""
    try:
        if os.path.exists("voice_command.txt"):
            with open("voice_command.txt", "r") as file:
                return file.read().strip()
        else:
            logger.warning("No registered voice command found.")
            return None
    except Exception as e:
        logger.error(f"Error loading voice command: {str(e)}")
        return None

# Function to display a message on the LCD in a continuous loop with scrolling effect
def display_message(message, stop_event):
    max_length = 16  # 16 characters for a 2x16 LCD

    # Ensure the message fits within the display
    if len(message) <= max_length:
        lcd.clear()
        lcd.write_string(message)
        sleep(2)
        return

    # Prepare to scroll the message
    message = message + "  "  # Add space for scrolling effect
    scroll_length = len(message)

    # Loop to display the message continuously
    while not stop_event.is_set():  # Check the stop event
        for i in range(scroll_length - max_length + 1):
            if stop_event.is_set():  # Check if we should stop
                break
            lcd.clear()
            lcd.write_string(message[i:i + max_length])
            sleep(0.5)  # Adjust the scroll speed as needed

def listen_for_command():
    """Listen for voice command."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("Listening for voice command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            logger.info(f"Voice command detected: {command}")
            return command
        except sr.UnknownValueError:
            logger.warning("Could not understand the audio")
            return None
        except sr.RequestError:
            logger.error("Could not request results; check internet connection")
            return None

# Add a global flag for buzzer activation
buzzer_activated = False

def sound_buzzer(duration=10):
    """Sound the buzzer for specified duration and set buzzer_activated to True."""
    global buzzer_activated
    buzzer_activated = True  # Set flag when buzzer is activated
    GPIO.output(PINS['buzzer'], GPIO.HIGH)
    sleep(duration)
    GPIO.output(PINS['buzzer'], GPIO.LOW)

# Function to handle fingerprint registration (enrollment)
def enroll_fingerprint():
    try:
        print('Waiting for finger for enrollment...')
        speak("Please place your finger on the sensor.")
        
        for _ in range(5):  # Max 5 attempts
            while not f.readImage():
                sleep(1)  # Delay for easier finger placement
                print("Place your finger on the sensor...")
            f.convertImage(0x01)
            result = f.searchTemplate()
            positionNumber = result[0]

            if positionNumber >= 0:
                print(f'Template already exists at position #{positionNumber}')
                speak('Fingerprint already registered.')
                return False

            print('Remove your finger.')
            speak('Remove your finger.')
            sleep(2)

            print('Place the same finger again...')
            speak('Place the same finger again.')

            for _ in range(5):  # Max 5 attempts
                while not f.readImage():
                    sleep(1)  # Delay for easier finger placement
                    print("Place your finger on the sensor...")
                f.convertImage(0x02)
                
                if f.compareCharacteristics() == 0:
                    raise Exception('Fingers do not match.')

                f.createTemplate()
                positionNumber = f.storeTemplate()
                print(f'Finger enrolled successfully at position #{positionNumber}')
                speak(f'Finger enrolled successfully at position {positionNumber}')
                return True
    except Exception as e:
        print('Error enrolling fingerprint!')
        print('Exception message: ' + str(e))
        speak('Error enrolling fingerprint.')
        return False

# Function to handle fingerprint verification
def verify_fingerprint():
    try:
        print('Waiting for finger...')
        while not f.readImage():
            sleep(0.5)
            print("Place your finger on the sensor...")

        f.convertImage(0x01)
        result = f.searchTemplate()
        positionNumber = result[0]

        if positionNumber >= 0:
            print(f'Fingerprint recognized at position #{positionNumber}')
            return True
        else:
            print('Fingerprint not recognized.')
            return False
    except Exception as e:
        print('Error verifying fingerprint!')
        print('Exception message: ' + str(e))
        return False

# Update the main loop to include saving and verifying the voice command
def main_loop():
    """Main program loop."""
    stop_event = threading.Event()
    voice_fail_count = 0
    fingerprint_fail_count = 0
    pin_fail_count = 0
    total_fail_count = 0
    


    # Initialize global variables
    global a_pressed, a_pressed_time
    a_pressed = False  # Flag for "A" press
    a_pressed_time = 0  # Time when "A" was pressed
    entered_password = ""  # Buffer for password input

    while True:
        # Check keypad
        key = read_keypad()
        if key:  # Check if a key is pressed
            if key == "A" and not a_pressed:
                message = "Hello, enter your PIN."
                speak(message)
            if a_pressed:
                if key == "OK":
                    # Check if the entered password matches
                    if entered_password == PASSWORD:
                        message = "Access granted!, Welcome!"
                        speak(message)
                        print("Access granted!")
                        update_lcd_display("Access granted!", "Welcome!")
                        open_lock()
                        update_notification("PIN access granted")
                        pin_fail_count = 0
                    else:
                        message = "Incorrect PIN!,Try again."
                        speak(message)
                        print("Incorrect PIN.")
                        update_lcd_display("Incorrect PIN!", "Try again.")
                        entered_password = ""  # Reset after checking
                        a_pressed = False
                        pin_fail_count += 1
                        update_notification("PIN access denied", True)
                        total_fail_count = voice_fail_count + fingerprint_fail_count + pin_fail_count
                        if total_fail_count >= 3:
                            sound_buzzer()
                            capture_and_email_intruder_image()
                        
                elif key == "*":
                    message = "Password cleared"
                    speak(message)
                    # Clear the entered password
                    entered_password = ""
                    print("Password cleared.")
                    update_lcd_display("PIN Cleared!", "")
                else:
                    # Append the key to the entered password
                    entered_password += key
                    print(f"Password so far: {entered_password}")
                    # Display the PIN on the LCD as it's being entered
                    update_lcd_display("Enter PIN:", "*" * len(entered_password))
            else:
                if key == "A":
                    # Activate PIN entry mode
                    a_pressed = True
                    entered_password = ""  # Reset PIN entry
                    print("A pressed. Enter your PIN.")
                    update_lcd_display("Enter your PIN", "")                                                                                                                                                                

        # Button 1: Register voice command
        if GPIO.input(PINS['button1']) == GPIO.LOW:
            message = "Hello, please register your voice password."
            speak(message)
            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()

            recorded_command = listen_for_command()
            if recorded_command:
                save_voice_command(recorded_command)  # Save the registered voice command
                stop_event.set()
                speak("Voice registered.")
                message = "Voice registered."
                update_notification("Voice registered")
                voice_fail_count = 0
            else:
                update_notification("Voice registration failed", True)
                stop_event.set()
                speak("Voice not clear, try again.")
                message = "Repeat and try again."

            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()


        # Button 2: Voice verification
        if GPIO.input(PINS['button2']) == GPIO.LOW:
            message = "Listening for voice verification."
            speak(message)
            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()

            registered_command = load_voice_command()  # Load the saved voice command
            if not registered_command:
                stop_event.set()
                speak("No voice command registered.")
                message = "No voice command registered."
                stop_event.clear()
                display_thread = threading.Thread(target=display_message, args=(message, stop_event))
                display_thread.start()
                continue

            command = listen_for_command()
            if command == registered_command:
                stop_event.set()
                speak("Access granted.")
                message = "Access granted."
                update_notification("Voice access granted")
                GPIO.output(PINS['solenoid'], GPIO.HIGH)
                sleep(5)
                GPIO.output(PINS['solenoid'], GPIO.LOW)
                voice_fail_count = 0
            else:
                stop_event.set()  # Stop the display
                speak("Access denied.")
                message = "Access denied."
                stop_event.clear()
                display_thread = threading.Thread(target=display_message, args=(message, stop_event))
                display_thread.start()
                voice_fail_count += 1
                update_notification("Voice access denied", True)
                total_fail_count = voice_fail_count + fingerprint_fail_count + pin_fail_count
                if total_fail_count >= 3:
                    sound_buzzer()
                    capture_and_email_intruder_image()

            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()

        sleep(0.1)  # Prevent CPU overuse

        # Button 4: Fingerprint verification
        if GPIO.input(PINS['button4']) == GPIO.LOW:
            message = "Please place your finger on the sensor."
            speak(message)
            stop_event.clear()  # Clear any previous stop event
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()  # Start displaying message in a separate thread

            if verify_fingerprint():
                stop_event.set()  # Stop the display
                speak("Fingerprint recognized. Access granted.")
                message = "Access granted."
                stop_event.clear()
                display_thread = threading.Thread(target=display_message, args=(message, stop_event))
                display_thread.start()
                update_notification("Fingerprint access granted")
                GPIO.output(PINS['solenoid'], GPIO.HIGH)
                sleep(5)
                GPIO.output(PINS['solenoid'], GPIO.LOW)
                fingerprint_fail_count = 0
            else:
                stop_event.set()  # Stop the display
                speak("Fingerprint not recognized. access denied")
                message = "Access denied."
                stop_event.clear()
                display_thread = threading.Thread(target=display_message, args=(message, stop_event))
                display_thread.start()
                fingerprint_fail_count += 1
                update_notification("Fingerprint access denied", True)
                total_fail_count = voice_fail_count + fingerprint_fail_count + pin_fail_count
                if total_fail_count >= 3:
                    sound_buzzer()
                    capture_and_email_intruder_image()

        # Button 5: Enroll fingerprint
        if GPIO.input(PINS['button5']) == GPIO.LOW:
            message = "Place your finger for enrollment."
            speak(message)
            stop_event.clear()  # Clear any previous stop event
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()  # Start displaying message in a separate thread
            
            if enroll_fingerprint():
                stop_event.set()  # Stop the display
                speak("Fingerprint enrolled successfully.")
                message = "Enroll successful"
                update_notification("Fingerprint enrolled successfully")
            else:
                update_notification("Fingerprint enrollment failed", True)
                stop_event.set()  # Stop the display
                speak("Fingerprint enrollment failed.")
                message = "Enroll failed."

            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()
            stop_event.set()  # Stop the display

        sleep(0.1)  # Prevent CPU overuse
        # Button 6: Open the lock
        if GPIO.input(PINS['button6']) == GPIO.LOW:
            message = "Lock is open."
            speak(message)
            stop_event.clear()
            display_thread = threading.Thread(target=display_message, args=(message, stop_event))
            display_thread.start()
            open_lock()


if __name__ == '__main__':
    try:
        setup()  # Ensure GPIO is set up before starting other components
        # Start main loop
        main_loop()
    except KeyboardInterrupt:
        GPIO.cleanup()
        logger.info("Program terminated.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        GPIO.cleanup()
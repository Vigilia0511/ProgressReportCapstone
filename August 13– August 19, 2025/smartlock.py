import RPi.GPIO as GPIO
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from RPLCD.i2c import CharLCD
import threading
from datetime import datetime
import logging
from picamera2 import Picamera2
import cv2
import time
import requests
import ssl
from typing import Optional
import re
import tempfile
from vosk import Model, KaldiRecognizer
import pyaudio
import wave
import mysql.connector
import face_recognition
import numpy as np
import random
from scipy.spatial import distance as dist

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define the GPIO pins for rows, columns, and button
ROW_PINS = [5, 27, 17, 4]
COL_PINS = [7, 8, 25, 18]
BUTTON_PIN = 20



# Keypad layout
KEYPAD = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'OK']
]

# Directory to save encoded faces
FACE_DIR = "saved_faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

# Constants for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Constants for mouth detection
MOUTH_AR_THRESH = 0.75

# Initialize the LCD
lcd = CharLCD('PCF8574', 0x27)

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Pin Definitions
PINS = {
    'button1': 24,  # Record unlock command
    'button6': 19,  # Manual unlock
    'button7': 9,   # Register face
    'solenoid': 6,
}


# Set up keypad GPIO
for row in ROW_PINS:
    GPIO.setup(row, GPIO.OUT)
    GPIO.output(row, GPIO.LOW)

for col in COL_PINS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration)
picam2.start()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.last_connection_attempt = 0
        self.reconnect_interval = 60  # Try to reconnect every 60 seconds
        
    def connect(self):
        """Connect to database."""
        try:
            self.connection = mysql.connector.connect(
                host="192.168.176.213",
                user="root",
                password="oneinamillion",
                database="Smartdb",
                connection_timeout=5,
                autocommit=True
            )
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("Database connected successfully")
            return True
        except Exception as e:
            logger.warning(f"Database connection failed: {str(e)}")
            self.connection = None
            self.cursor = None
            return False
    
    def ensure_connection(self):
        """Ensure database connection is available when online."""
        current_time = time.time()
        
        # Only try to reconnect if we're online and enough time has passed
        if (system_mode.is_online and 
            not self.connection and 
            current_time - self.last_connection_attempt > self.reconnect_interval):
            
            self.last_connection_attempt = current_time
            if self.connect():
                speak("Database connection restored")
                update_lcd_display("Database", "Connected")
                sleep(1)
    
    def is_connected(self):
        """Check if database is connected."""
        try:
            if self.connection:
                self.connection.ping(reconnect=True, attempts=1, delay=0)
                return True
        except:
            self.connection = None
            self.cursor = None
        return False

class SystemMode:
    def __init__(self):
        self.is_online = False
        self.last_network_check = 0
        self.network_check_interval = 30  # Check every 30 seconds
        self.mode_change_callbacks = []
        
    def check_network_status(self):
        """Check if network is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def update_mode(self):
        """Update system mode based on network availability."""
        current_time = time.time()
        
        # Only check network periodically to avoid constant checking
        if current_time - self.last_network_check > self.network_check_interval:
            self.last_network_check = current_time
            new_status = self.check_network_status()
            
            # Mode change detected
            if new_status != self.is_online:
                old_mode = "Online" if self.is_online else "Offline"
                new_mode = "Online" if new_status else "Offline"
                
                self.is_online = new_status
                logger.info(f"System mode changed: {old_mode} -> {new_mode}")
                
                # Notify about mode change
                try:
                    speak(f"System switched to {new_mode.lower()} mode")
                    update_lcd_display(f"{new_mode} Mode", "Active")
                    sleep(2)
                except:
                    pass
                
                # Execute callbacks
                for callback in self.mode_change_callbacks:
                    try:
                        callback(self.is_online)
                    except Exception as e:
                        logger.error(f"Mode change callback error: {str(e)}")
    
    def add_mode_change_callback(self, callback):
        """Add callback to be executed when mode changes."""
        self.mode_change_callbacks.append(callback)

# Initialize system mode manager
system_mode = SystemMode()

# Initialize database manager
db_manager = DatabaseManager()
db_manager.connect()

# For backward compatibility
db_connection = db_manager.connection
db_cursor = db_manager.cursor

# Set the correct password
PASSWORD = "1234"

# Authentication state tracking
class AuthenticationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.authenticated_methods = set()
        self.reset_timer = None
        self.is_unlocking = False
        self.unlock_complete = False
        
    def add_authentication(self, method):
        with self.lock:
            if self.is_unlocking:  # Prevent adding auth during unlock
                return
                
            self.authenticated_methods.add(method)
            logger.info(f"Authentication method '{method}' verified. Total: {len(self.authenticated_methods)}")
            
            # Reset timer - clear authentications after 30 seconds
            if self.reset_timer:
                self.reset_timer.cancel()
            self.reset_timer = threading.Timer(30.0, self.reset_authentications)
            self.reset_timer.start()
            
            # Check if we have two different authentication methods
            if len(self.authenticated_methods) >= 2:
                # Start unlock in separate thread to prevent blocking
                unlock_thread = threading.Thread(target=self.unlock_door, daemon=True)
                unlock_thread.start()
                
    def reset_authentications(self):
        with self.lock:
            if not self.is_unlocking:
                self.authenticated_methods.clear()
                self.unlock_complete = False
                logger.info("Authentication state reset - timeout")
            
    def unlock_door(self):
        with self.lock:
            if self.is_unlocking:  # Prevent multiple unlock attempts
                return
            self.is_unlocking = True
            
        try:
            logger.info("sequential authentication successful - unlocking door")
            
            # Perform unlock operations
            speak("sequential authentication successful. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            
            # Activate solenoid
            GPIO.output(PINS['solenoid'], GPIO.HIGH)
            sleep(5)  # Keep door unlocked for 5 seconds
            GPIO.output(PINS['solenoid'], GPIO.LOW)
            
            logger.info("Door lock cycle completed")
            
        except Exception as e:
            logger.error(f"Error during unlock: {str(e)}")
        finally:
            # Reset state after unlock
            with self.lock:
                self.authenticated_methods.clear()
                self.is_unlocking = False
                self.unlock_complete = True
                if self.reset_timer:
                    self.reset_timer.cancel()
            
            # Update display back to ready state after a brief delay
            sleep(2)
            update_lcd_display("Smart Lock Ready", "2FA Required")
            logger.info("System ready for next authentication")

# Global authentication state
auth_state = AuthenticationState()

class NotificationManager:
    def __init__(self):
        self.offline_log_file = "offline_logs.txt"
        
    def log_notification(self, user_id, notify):
        """Log notification with automatic online/offline handling."""
        
        # Update database connection status
        db_manager.ensure_connection()
        
        # Try database logging if online and connected
        if system_mode.is_online and db_manager.is_connected():
            try:
                query = "INSERT INTO logs (user_id, notify, timestamp) VALUES (%s, %s, NOW())"
                db_manager.cursor.execute(query, (user_id, notify))
                db_manager.connection.commit()
                logger.info(f"Logged to database: {notify}")
                return
            except Exception as e:
                logger.warning(f"Database logging failed: {str(e)}, using offline log")
        
        # Offline logging fallback
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode = "ONLINE" if system_mode.is_online else "OFFLINE"
            with open(self.offline_log_file, "a") as log_file:
                log_file.write(f"[{mode}] {timestamp} - User: {user_id} - Event: {notify}\n")
            logger.info(f"Logged offline: {notify}")
        except Exception as e:
            logger.error(f"Offline logging failed: {str(e)}")

# Initialize notification manager
notification_manager = NotificationManager()

# Initialize user_id variable (used in logging)
user_id = "default_user"

# Face Recognition Functions
def get_next_face_id():
    """Get the next available face ID (1, 2, 3, ...)."""
    existing_files = os.listdir(FACE_DIR)
    if not existing_files:
        return 1
    existing_ids = [int(f.split(".")[0]) for f in existing_files if f.endswith(".npy")]
    return max(existing_ids) + 1 if existing_ids else 1

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR) to detect blinks."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate the mouth aspect ratio (MAR) to detect mouth open."""
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (2.0 * D)
    return mar

def capture_and_save_face():
    """Capture and save face for registration."""
    speak("Position yourself in front of the camera for face registration. Capturing in 3 seconds.")
    update_lcd_display("Face Registration", "Position yourself")
    time.sleep(3)
    
    frame = picam2.capture_array()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        speak("No face detected. Please try again.")
        update_lcd_display("No Face Detected", "Try Again")
        return False

    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

    # Load existing face encodings
    known_face_encodings = []
    known_face_ids = []
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            face_id = os.path.splitext(file)[0]
            face_encoding_saved = np.load(f"{FACE_DIR}/{file}")
            known_face_encodings.append(face_encoding_saved)
            known_face_ids.append(face_id)

    # Check if face already exists
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    if True in matches:
        first_match_index = matches.index(True)
        face_id = known_face_ids[first_match_index]
        speak("Face already registered, updating encoding")
        update_lcd_display("Face Updated", f"ID: {face_id}")
    else:
        face_id = get_next_face_id()
        speak("New face registered successfully")
        update_lcd_display("Face Registered", f"ID: {face_id}")

    # Save face encoding
    np.save(f"{FACE_DIR}/{face_id}.npy", face_encoding)
    notification_manager.log_notification(user_id, "Face registered")
    logger.info(f"Face saved as {face_id}.npy")
    return True

def verify_face():
    """Verify face with liveness detection."""
    speak("Position yourself for face verification. Perform blink and mouth movement.")
    update_lcd_display("Face Verification", "Blink & Move Mouth")
    
    # Load saved face encodings
    known_face_encodings = []
    known_face_ids = []
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            face_id = os.path.splitext(file)[0]
            face_encoding = np.load(f"{FACE_DIR}/{file}")
            known_face_encodings.append(face_encoding)
            known_face_ids.append(face_id)

    if not known_face_encodings:
        speak("No faces registered. Please register a face first.")
        update_lcd_display("No Faces", "Register First")
        return False

    blink_counter = 0
    blink_detected = False
    mouth_open_detected = False
    start_time = time.time()

    while time.time() - start_time < 15:  # Reduced timeout to 15 seconds
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            if True in matches:
                first_match_index = matches.index(True)
                face_id = known_face_ids[first_match_index]

                # Liveness detection
                landmarks = face_recognition.face_landmarks(rgb_frame, [(top, right, bottom, left)])
                if landmarks:
                    landmarks = landmarks[0]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    mouth = landmarks["top_lip"] + landmarks["bottom_lip"]

                    # Calculate EAR and MAR
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(mouth)

                    # Blink detection
                    if ear < EYE_AR_THRESH:
                        blink_counter += 1
                    else:
                        if blink_counter >= EYE_AR_CONSEC_FRAMES:
                            blink_detected = True
                        blink_counter = 0

                    # Mouth open detection
                    if mar > MOUTH_AR_THRESH:
                        mouth_open_detected = True

                    # Check if liveness tests passed
                    if blink_detected and mouth_open_detected:
                        speak(f"Face verified for user {face_id}")
                        update_lcd_display("Face Verified", f"User: {face_id}")
                        notification_manager.log_notification(user_id, "Face access granted")
                        auth_state.add_authentication("face")
                        return True
            else:
                # Unknown face detected
                speak("Unknown face detected. Access denied.")
                update_lcd_display("Unknown Face", "Access Denied")
                notification_manager.log_notification(user_id, "Face access denied")
                sound_buzzer(3)
                save_intruder_image()
                return False

        time.sleep(0.1)

    speak("Face verification failed. Liveness checks not passed.")
    update_lcd_display("Verification", "Failed")
    notification_manager.log_notification(user_id, "Face verification failed")
    return False

# Declare forward references for functions used in the above section
def speak(message):
    """Forward declaration - implemented below"""
    pass

def update_lcd_display(line1, line2=""):
    """Forward declaration - implemented below"""
    pass

def sound_buzzer(duration=3):
    """Forward declaration - implemented below"""
    pass

def save_intruder_image():
    """Forward declaration - implemented below"""
    pass

#//JUST FIX THE BELOW CODE OF THIS COMMENT JUST GENERATE THE FIX CODE OF THE CODE FROM HERE TO THE BELLOW ONLY ALL OF THE CODE HERE TO THE BUTTOM 

def save_intruder_image():
    """Save intruder image to database."""
    try:
        frame = picam2.capture_array()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"/tmp/intruder_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        save_intruder_image_to_db(image_path)
        os.remove(image_path)
    except Exception as e:
        logger.error(f"Error saving intruder image: {str(e)}")

def save_intruder_image_to_db(image_path):
    """Save image with automatic online/offline handling."""
    
    # Update database connection status
    db_manager.ensure_connection()
    
    # Try database storage if online and connected
    if system_mode.is_online and db_manager.is_connected():
        try:
            with open(image_path, 'rb') as file:
                binary_data = file.read()
            
            insert_query = "INSERT INTO images (image) VALUES (%s)"
            db_manager.cursor.execute(insert_query, (binary_data,))
            db_manager.connection.commit()
            logger.info("Intruder image saved to database")
            return
        except Exception as e:
            logger.warning(f"Database image save failed: {str(e)}, using local storage")
    
    # Offline storage fallback
    try:
        offline_dir = "offline_images"
        if not os.path.exists(offline_dir):
            os.makedirs(offline_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "ONLINE" if system_mode.is_online else "OFFLINE"
        offline_path = f"{offline_dir}/intruder_{mode}_{timestamp}.jpg"
        
        import shutil
        shutil.copy2(image_path, offline_path)
        logger.info(f"Intruder image saved locally: {offline_path}")
        
    except Exception as e:
        logger.error(f"Local image save failed: {str(e)}")

def check_offline_dependencies():
    """Check if offline dependencies are available."""
    issues = []
    
    # Check Vosk model
    if not os.path.exists("vosk-model"):
        issues.append("Vosk model not found - offline speech recognition unavailable")
    
    # Check pico2wave
    if os.system("which pico2wave > /dev/null 2>&1") != 0:
        issues.append("pico2wave not found - install libttspico-utils")
    
    # Check espeak
    if os.system("which espeak > /dev/null 2>&1") != 0:
        issues.append("espeak not found - install espeak")
    
    if issues:
        logger.warning("Offline dependencies issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        speak("Some offline features may not work properly")
    else:
        logger.info("All offline dependencies available")
    
    return len(issues) == 0

# Voice Recognition Functions
def save_voice_command(command):
    """Save voice command to file."""
    try:
        with open("voice_command.txt", "w") as file:
            file.write(command)
        logger.info("Voice command saved successfully")
    except Exception as e:
        logger.error(f"Error saving voice command: {str(e)}")

def load_voice_command():
    """Load voice command from file."""
    try:
        if os.path.exists("voice_command.txt"):
            with open("voice_command.txt", "r") as file:
                return file.read().strip()
        return None
    except Exception as e:
        logger.error(f"Error loading voice command: {str(e)}")
        return None

def listen_for_command():
    """Listen for voice command with automatic online/offline switching."""
    
    # Try online recognition first if available
    if system_mode.is_online:
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            with sr.Microphone() as source:
                logger.info("Listening for voice command (online)...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                command = recognizer.recognize_google(audio)
                logger.info(f"Voice command detected (online): {command}")
                return command.lower().strip()
                
        except sr.UnknownValueError:
            logger.warning("Could not understand audio (online)")
        except sr.RequestError:
            logger.warning("Online speech recognition failed, switching to offline")
            system_mode.is_online = False  # Force offline mode for this session
        except Exception as e:
            logger.warning(f"Online voice recognition error: {str(e)}")
    
    # Offline recognition fallback
    try:
        # Check if Vosk model exists
        if not os.path.exists("vosk-model"):
            logger.error("Vosk model not found. Please install offline model.")
            speak("Offline voice recognition not available")
            return None
            
        import vosk
        import pyaudio
        import json
        
        model = vosk.Model("vosk-model")
        recognizer = vosk.KaldiRecognizer(model, 16000)
        
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                         input=True, frames_per_buffer=8192)
        stream.start_stream()
        
        logger.info("Listening for voice command (offline)...")
        
        timeout = time.time() + 10
        
        while time.time() < timeout:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                command = result.get('text', '').lower().strip()
                if command:
                    logger.info(f"Voice command detected (offline): {command}")
                    stream.stop_stream()
                    stream.close()
                    mic.terminate()
                    return command
        
        # Get final result
        result = json.loads(recognizer.FinalResult())
        command = result.get('text', '').lower().strip()
        
        stream.stop_stream()
        stream.close()
        mic.terminate()
        
        if command:
            logger.info(f"Voice command detected (offline): {command}")
            return command
        else:
            logger.warning("No voice command detected (offline)")
            return None
            
    except Exception as e:
        logger.error(f"Offline voice recognition error: {str(e)}")
        return None

# Keypad Functions
def read_keypad():
    """Read keypad input with improved debouncing."""
    for row_index, row_pin in enumerate(ROW_PINS):
        GPIO.output(row_pin, GPIO.HIGH)
        for col_index, col_pin in enumerate(COL_PINS):
            if GPIO.input(col_pin) == GPIO.HIGH:
                time.sleep(0.05)
                if GPIO.input(col_pin) == GPIO.HIGH:
                    key = KEYPAD[row_index][col_index]
                    # Wait for key release with timeout
                    timeout = time.time() + 2.0
                    while GPIO.input(col_pin) == GPIO.HIGH and time.time() < timeout:
                        time.sleep(0.01)
                    GPIO.output(row_pin, GPIO.LOW)
                    return key
        GPIO.output(row_pin, GPIO.LOW)
    return None

def speak(message):
    """Text-to-speech with automatic online/offline switching."""
    
    # Try online TTS first if available
    if system_mode.is_online:
        try:
            from gtts import gTTS
            tts = gTTS(text=message, lang='en')
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                tts.save(fp.name)
                result = os.system(f"mpg321 {fp.name} > /dev/null 2>&1")
                if result == 0:
                    return  # Success with online TTS
        except Exception as e:
            logger.warning(f"Online TTS failed: {str(e)}, switching to offline")
    
    # Offline TTS fallback
    try:
        # Try pico2wave first (better quality)
        temp_file = "/tmp/output.wav"
        command = f'pico2wave -w {temp_file} "{message}" && aplay {temp_file} > /dev/null 2>&1 && rm {temp_file}'
        result = os.system(command)
        
        if result == 0:
            return  # Success with pico2wave
        
        # Fallback to espeak
        os.system(f'espeak "{message}" > /dev/null 2>&1')
        
    except Exception as e:
        logger.error(f"All TTS methods failed: {str(e)}")

def update_lcd_display(line1, line2=""):
    """Update LCD display with error handling."""
    try:
        lcd.clear()
        lcd.write_string(line1[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:16])
    except Exception as e:
        logger.error(f"LCD update error: {str(e)}")


def display_message(message, stop_event):
    """Display scrolling message on LCD."""
    max_length = 16
    if len(message) <= max_length:
        lcd.clear()
        lcd.write_string(message)
        sleep(2)
        return
    
    message = message + "  "
    scroll_length = len(message)
    
    while not stop_event.is_set():
        for i in range(scroll_length - max_length + 1):
            if stop_event.is_set():
                break
            lcd.clear()
            lcd.write_string(message[i:i + max_length])
            sleep(0.5)

# Button debouncing function
def is_button_pressed(pin, debounce_time=0.1):
    """Check if button is pressed with debouncing."""
    if GPIO.input(pin) == GPIO.LOW:
        time.sleep(debounce_time)
        if GPIO.input(pin) == GPIO.LOW:
            # Wait for button release with timeout
            timeout = time.time() + 2.0
            while GPIO.input(pin) == GPIO.LOW and time.time() < timeout:
                time.sleep(0.01)
            return True
    return False


def try_authentication(method, total_failures, max_failures):
    """Attempt authentication with the specified method, return (success, failures)."""
    if method == "pin":
        speak("Enter your pin")
        update_lcd_display("Enter PIN:", "")
        a_pressed = True
        entered_password = ""
        last_key_time = time.time()
        key_timeout = 5.0
        
        while a_pressed:
            key = read_keypad()
            current_time = time.time()
            
            if key:
                last_key_time = current_time
                logger.info(f"Key pressed: {key}")
                
                if key == "OK":
                    if entered_password == PASSWORD:
                        speak("PIN verified. Door unlocked.")
                        update_lcd_display("Access Granted!", "Door Unlocked")
                        GPIO.output(PINS['solenoid'], GPIO.HIGH)
                        sleep(5)
                        GPIO.output(PINS['solenoid'], GPIO.LOW)
                        notification_manager.log_notification(user_id, "PIN access granted")
                        return True, total_failures
                    else:
                        speak("Incorrect PIN")
                        update_lcd_display("Incorrect PIN", "Try again")
                        total_failures += 1
                        notification_manager.log_notification(user_id, "PIN access denied")
                        a_pressed = False
                elif key == "*":
                    entered_password = ""
                    update_lcd_display("PIN Cleared", "")
                elif key in "0123456789":
                    entered_password += key
                    update_lcd_display("Enter PIN:", "*" * len(entered_password))
                
                if total_failures >= max_failures:
                    return False, total_failures
                
            if (current_time - last_key_time) > key_timeout:
                a_pressed = False
                entered_password = ""
                update_lcd_display("PIN Timeout", "Try again")
                total_failures += 1
                notification_manager.log_notification(user_id, "PIN access denied - timeout")
            
            sleep(0.05)
        
        return False, total_failures
    
    elif method == "voice":
        registered_command = load_voice_command()
        if not registered_command:
            speak("No voice command registered")
            update_lcd_display("No Voice", "Registered")
            sleep(2)
            return False, total_failures + 1
        
        speak("Verify your voice password")
        update_lcd_display("Voice Verify", "Speak now")
        command = listen_for_command()
        if command and command == registered_command:
            speak("Voice verified. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            GPIO.output(PINS['solenoid'], GPIO.HIGH)
            sleep(5)
            GPIO.output(PINS['solenoid'], GPIO.LOW)
            notification_manager.log_notification(user_id, "Voice access granted")
            return True, total_failures
        else:
            speak("Voice not recognized")
            update_lcd_display("Voice Failed", "Try again")
            notification_manager.log_notification(user_id, "Voice access denied")
            return False, total_failures + 1
    
    elif method == "face":
        if verify_face():
            speak("Face verified. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            GPIO.output(PINS['solenoid'], GPIO.HIGH)
            sleep(5)
            GPIO.output(PINS['solenoid'], GPIO.LOW)
            return True, total_failures
        else:
            logger.info(f"Face verification failed. Total failures: {total_failures + 1}/{max_failures}")
            return False, total_failures + 1
    
    return False, total_failures

def main_loop():
    """Main program loop with sinsequential authentication and random fallback."""
    # Unified failure counter for all authentication methods
    total_failures = 0
    max_failures = 3
    
    # Authentication methods
    auth_methods = ["pin", "voice", "face"]
    
    update_lcd_display("Smart Lock Ready", "Select Method")

    # Initialize mode and display status
    system_mode.update_mode()
    mode_text = "Online Mode" if system_mode.is_online else "Offline Mode"
    update_lcd_display("Smart Lock Ready", mode_text)
    
    # Add mode change callback
    def on_mode_change(is_online):
        mode_text = "Online" if is_online else "Offline"
        update_lcd_display(f"{mode_text} Mode", "Active")
        notification_manager.log_notification(user_id, f"System switched to {mode_text} mode")
    
    system_mode.add_mode_change_callback(on_mode_change)
    
    while True:
        try:
            system_mode.update_mode()
            
            # Check keypad for initiating authentication
            key = read_keypad()
            if key:
                logger.info(f"Key pressed: {key}")
                
                # Map keypad input to authentication method
                key_to_method = {
                    "A": "pin",
                    "1": "voice",
                    "2": "face",
                }
                
                if key in key_to_method:
                    method = key_to_method[key]
                    remaining_methods = auth_methods.copy()
                    remaining_methods.remove(method)  # Remove the chosen method
                    current_methods = [method]  # Start with the chosen method
                    
                    while current_methods:
                        current_method = current_methods[0]
                        success, total_failures = try_authentication(current_method, total_failures, max_failures)
                        if success:
                            total_failures = 0  # Reset on success
                            sleep(2)
                            break
                        else:
                            if total_failures >= max_failures:
                                save_intruder_image()
                                sound_buzzer(8)
                                total_failures = 0
                                sleep(5)
                                break
                            # Try a random method from remaining ones
                            if remaining_methods:
                                next_method = random.choice(remaining_methods)
                                remaining_methods.remove(next_method)
                                current_methods = [next_method]
                            else:
                                speak("All authentication methods failed")
                                update_lcd_display("All Methods", "Failed")
                                sleep(2)
                                break
                    sleep(2)
                
                # Handle invalid keypad input
                elif key == "OK":
                    update_lcd_display("Invalid Input", "Try again")
                    sleep(2)
            
            # Check buttons for registration and manual unlock
            # Button 1: Register voice command
            if is_button_pressed(PINS['button1']):
                speak("Register your voice password")
                update_lcd_display("Voice Register", "Speak now")
                
                command = listen_for_command()
                if command:
                    save_voice_command(command)
                    speak("Voice registered successfully")
                    update_lcd_display("Voice", "Registered")
                    notification_manager.log_notification(user_id, "Voice registered")
                else:
                    speak("Voice registration failed")
                    update_lcd_display("Registration", "Failed")
                sleep(2)
    
if __name__ == '__main__':
    try:
        check_offline_dependencies()
        server_thread.start()
        
        logger.info("Smart Lock System with sequential Authentication Starting...")
        speak("Smart lock system ready. sequential authentication required.")
        
        # Start main loop
        main_loop()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        GPIO.cleanup()
        lcd.clear()
        lcd.backlight_enabled = False
        logger.info("System shutdown complete")
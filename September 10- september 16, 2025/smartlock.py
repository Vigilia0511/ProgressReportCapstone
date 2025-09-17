import RPi.GPIO as GPIO
from time import sleep
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
from RPLCD.i2c import CharLCD
from pyfingerprint.pyfingerprint import PyFingerprint
import threading
import multiprocessing
from datetime import datetime
from flask import Flask, request, jsonify, Response
import logging
from picamera2 import Picamera2
import cv2
import time
import requests
import ssl
from typing import Optional
import mysql.connector
import face_recognition
import numpy as np
import random
from scipy.spatial import distance as dist
import json
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import queue
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

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
    'button5': 1,   # Register fingerprint
    'button6': 13,  # Manual unlock
    'button7': 20,  # Register face
    'solenoid': 6,
    'buzzer': 22,
}

# Setup GPIO pins
for pin in PINS.values():
    if pin in [PINS['solenoid'], PINS['buzzer']]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    else:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Set up keypad GPIO
for row in ROW_PINS:
    GPIO.setup(row, GPIO.OUT)
    GPIO.output(row, GPIO.LOW)

for col in COL_PINS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Initialize fingerprint sensor
try:
    f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)
    if not f.verifyPassword():
        raise ValueError('The given fingerprint sensor password is incorrect!')
except Exception as e:
    logger.error(f'Fingerprint sensor initialization failed: {str(e)}')
    exit(1)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.preview_configuration)
picam2.start()

# Global video streaming control
class VideoStreamManager:
    def __init__(self):
        self.streaming = False
        self.stream_lock = threading.Lock()
        self.frame_cache = None
        self.cache_time = 0
        self.cache_duration = 0.125  # Cache frame for 125ms (8 FPS)
        self.frame_thread = None
        self.stop_event = threading.Event()
    
    def start_streaming(self):
        with self.stream_lock:
            if not self.streaming:
                self.streaming = True
                self.stop_event.clear()
                self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
                self.frame_thread.start()
                logger.info("Video streaming started (threading)")
    
    def stop_streaming(self):
        with self.stream_lock:
            if self.streaming:
                self.streaming = False
                self.stop_event.set()
                if hasattr(self, 'frame_thread') and self.frame_thread:
                    self.frame_thread.join(timeout=2.0)
                    self.frame_thread = None
                logger.info("Video streaming stopped (threading)")
    
    def is_streaming(self):
        with self.stream_lock:
            return self.streaming
    
    def get_frame(self):
        """Get cached frame to reduce camera load"""
        with self.stream_lock:
            if self.frame_cache is None or time.time() - self.cache_time > self.cache_duration:
                try:
                    frame = picam2.capture_array()
                    if frame is not None:
                        self.frame_cache = frame
                        self.cache_time = time.time()
                    return frame
                except Exception as e:
                    logger.error(f"Frame capture error: {str(e)}")
                    return None
            return self.frame_cache
    
    def _capture_frames(self):
        """Dedicated thread for frame capture"""
        while not self.stop_event.is_set():
            frame = picam2.capture_array()
            with self.stream_lock:
                self.frame_cache = frame
                self.cache_time = time.time()
            time.sleep(0.125)  # Limit to ~8 FPS to reduce load

video_manager = VideoStreamManager()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.last_connection_attempt = 0
        self.reconnect_interval = 60
        
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host="192.168.1.149",
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
        current_time = time.time()
        if (system_mode.is_online and 
            not self.connection and 
            current_time - self.last_connection_attempt > self.reconnect_interval):
            self.last_connection_attempt = current_time
            if self.connect():
                speak("Database connection restored")
                update_lcd_display("Database", "Connected")
                sleep(1)
    
    def is_connected(self):
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
        self.network_check_interval = 30
        self.mode_change_callbacks = []
        
    def check_network_status(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def update_mode(self):
        current_time = time.time()
        if current_time - self.last_network_check > self.network_check_interval:
            self.last_network_check = current_time
            new_status = self.check_network_status()
            if new_status != self.is_online:
                old_mode = "Online" if self.is_online else "Offline"
                new_mode = "Online" if new_status else "Offline"
                self.is_online = new_status
                logger.info(f"System mode changed: {old_mode} -> {new_mode}")
                try:
                    speak(f"System switched to {new_mode.lower()} mode")
                    update_lcd_display(f"{new_mode} Mode", "Active")
                    sleep(2)
                except:
                    pass
                for callback in self.mode_change_callbacks:
                    try:
                        callback(self.is_online)
                    except Exception as e:
                        logger.error(f"Mode change callback error: {str(e)}")
    
    def add_mode_change_callback(self, callback):
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
            if self.is_unlocking:
                return
            self.authenticated_methods.add(method)
            logger.info(f"Authentication method '{method}' verified. Total: {len(self.authenticated_methods)}")
            if self.reset_timer:
                self.reset_timer.cancel()
            self.reset_timer = threading.Timer(30.0, self.reset_authentications)
            self.reset_timer.start()
            if len(self.authenticated_methods) >= 2:
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
            if self.is_unlocking:
                return
            self.is_unlocking = True
        try:
            logger.info("Two-factor authentication successful - unlocking door")
            speak("Two factor authentication successful. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            def solenoid_callback(success):
                if success:
                    logger.info("Solenoid unlock cycle completed")
                    notification_manager.log_notification(user_id, "Door unlocked via 2FA")
                else:
                    logger.error("Failed to complete solenoid unlock cycle")
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
            sleep(5)
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
            logger.info("Door lock cycle completed")
        except Exception as e:
            logger.error(f"Error during unlock: {str(e)}")
        finally:
            with self.lock:
                self.authenticated_methods.clear()
                self.is_unlocking = False
                self.unlock_complete = True
                if self.reset_timer:
                    self.reset_timer.cancel()
            sleep(2)
            update_lcd_display("Smart Lock Ready", "2FA Required")
            logger.info("System ready for next authentication")

# Global authentication state
auth_state = AuthenticationState()

# Safe GPIO operation wrapper
def safe_gpio_operation(operation, max_retries=5, retry_delay=0.05):
    for attempt in range(max_retries):
        try:
            operation()
            return True
        except Exception as e:
            logger.warning(f"GPIO operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("GPIO operation failed after all retries")
                return False
    return False

# Initialize solenoid operation queue
solenoid_queue = queue.Queue()

# Health check for solenoid worker
solenoid_worker_alive = threading.Event()

def solenoid_worker():
    solenoid_worker_alive.set()
    while True:
        try:
            operation, callback = solenoid_queue.get()
            success = safe_gpio_operation(operation)
            if callback:
                callback(success)
            solenoid_queue.task_done()
        except Exception as e:
            logger.error(f"Solenoid worker error: {str(e)}")

@app.route('/solenoid_health')
def solenoid_health():
    alive = solenoid_worker_alive.is_set()
    queue_size = solenoid_queue.qsize()
    return jsonify({"worker_alive": alive, "queue_size": queue_size})

class NotificationManager:
    def __init__(self):
        self.offline_log_file = "offline_logs.txt"
        
    def log_notification(self, user_id, notify):
        db_manager.ensure_connection()
        if system_mode.is_online and db_manager.is_connected():
            try:
                query = "INSERT INTO logs (user_id, notify, timestamp) VALUES (%s, %s, NOW())"
                db_manager.cursor.execute(query, (user_id, notify))
                db_manager.connection.commit()
                logger.info(f"Logged to database: {notify}")
                return
            except Exception as e:
                logger.warning(f"Database logging failed: {str(e)}, using offline log")
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
    existing_files = os.listdir(FACE_DIR)
    if not existing_files:
        return 1
    existing_ids = [int(f.split(".")[0]) for f in existing_files if f.endswith(".npy")]
    return max(existing_ids) + 1 if existing_ids else 1

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
    D = dist.euclidean(mouth[0], mouth[4])
    mar = (A + B + C) / (2.0 * D)
    return mar

def enhance_image_for_dark(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
    return enhanced

def set_camera_for_low_light():
    try:
        picam2.set_controls({"AnalogueGain": 16.0, "ExposureTime": 1000000})
    except Exception as e:
        logger.warning(f"Camera low-light settings failed: {str(e)}")

def capture_and_save_face():
    speak("Position yourself in front of the camera for face registration. Capturing in 3 seconds.")
    update_lcd_display("Face Registration", "Position yourself")
    time.sleep(3)
    frame = video_manager.get_frame()
    if frame is None:
        speak("Camera error. Please try again.")
        update_lcd_display("Camera Error", "Try Again")
        return False
    frame = enhance_image_for_dark(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        speak("No face detected. Please try again.")
        update_lcd_display("No Face Detected", "Try Again")
        return False
    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    known_face_encodings = []
    known_face_ids = []
    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            face_id = os.path.splitext(file)[0]
            face_encoding_saved = np.load(f"{FACE_DIR}/{file}")
            known_face_encodings.append(face_encoding_saved)
            known_face_ids.append(face_id)
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
    np.save(f"{FACE_DIR}/{face_id}.npy", face_encoding)
    notification_manager.log_notification(user_id, "Face registered")
    logger.info(f"Face saved as {face_id}.npy")
    return True

def verify_face():
    speak("Position yourself for face verification. Capturing in 3 seconds.")
    update_lcd_display("Face Verification", "Position yourself")
    time.sleep(3)
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
    frame = video_manager.get_frame()
    if frame is None:
        speak("Camera error. Access denied.")
        update_lcd_display("Camera Error", "Access Denied")
        notification_manager.log_notification(user_id, "Face access denied - camera error")
        return False
    frame = enhance_image_for_dark(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if len(face_locations) == 0:
        speak("No face detected. Access denied.")
        update_lcd_display("No Face Detected", "Access Denied")
        notification_manager.log_notification(user_id, "Face access denied - no face detected")
        save_intruder_image()
        return False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            face_id = known_face_ids[first_match_index]
            speak(f"Face verified for user {face_id}")
            update_lcd_display("Face Verified", f"User: {face_id}")
            notification_manager.log_notification(user_id, "Face access granted")
            auth_state.add_authentication("face")
            return True
    speak("Unknown face detected. Access denied.")
    update_lcd_display("Unknown Face", "Access Denied")
    notification_manager.log_notification(user_id, "Face access denied")
    save_intruder_image()
    return False

def save_intruder_image():
    try:
        frame = video_manager.get_frame()
        if frame is None:
            logger.error("Cannot capture intruder image - camera error")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"/tmp/intruder_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        save_intruder_image_to_db(image_path)
        os.remove(image_path)
    except Exception as e:
        logger.error(f"Error saving intruder image: {str(e)}")

def save_intruder_image_to_db(image_path):
    db_manager.ensure_connection()
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
    issues = []
    if not os.path.exists("vosk-model"):
        issues.append("Vosk model not found - offline speech recognition unavailable")
    if os.system("which pico2wave > /dev/null 2>&1") != 0:
        issues.append("pico2wave not found - install libttspico-utils")
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

def save_voice_command(command):
    try:
        with open("voice_command.txt", "w") as file:
            file.write(command)
        logger.info("Voice command saved successfully")
    except Exception as e:
        logger.error(f"Error saving voice command: {str(e)}")

def load_voice_command():
    try:
        if os.path.exists("voice_command.txt"):
            with open("voice_command.txt", "r") as file:
                return file.read().strip()
        return None
    except Exception as e:
        logger.error(f"Error loading voice command: {str(e)}")
        return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def enroll_voice_resemblyzer():
    encoder = VoiceEncoder()
    profiles = {}
    for i in range(1, 11):
        profile_path = f"voice_profile_{i}_resemblyzer.npy"
        if os.path.exists(profile_path):
            profiles[i] = np.load(profile_path)
    speak("Please speak naturally for 15 seconds to enroll your voice.")
    fs = 16000
    duration = 15
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    wav = np.squeeze(audio)
    wav = preprocess_wav(wav, source_sr=fs)
    embed = encoder.embed_utterance(wav)
    matched_id = None
    best_sim = 0
    for user_id, profile in profiles.items():
        sim = np.dot(embed, profile) / (np.linalg.norm(embed) * np.linalg.norm(profile))
        logger.info(f"Resemblyzer similarity with ID {user_id}: {sim:.2f}")
        if sim > 0.92 and sim > best_sim:
            matched_id = user_id
            best_sim = sim
    if matched_id:
        speak(f"Voice already registered as user {matched_id}. Profile updated.")
        np.save(f"voice_profile_{matched_id}_resemblyzer.npy", embed)
        notification_manager.log_notification(f"user{matched_id}", "Voice profile updated (Resemblyzer)")
        return True
    else:
        for i in range(1, 11):
            if i not in profiles:
                np.save(f"voice_profile_{i}_resemblyzer.npy", embed)
                speak(f"Voice enrolled successfully as user {i}.")
                notification_manager.log_notification(f"user{i}", "Voice profile enrolled (Resemblyzer)")
                return True
        speak("Maximum number of users (10) reached. Enrollment failed.")
        return False

def listen_for_command_resemblyzer():
    encoder = VoiceEncoder()
    profiles = {}
    for file in os.listdir("."):
        if file.startswith("voice_profile_") and file.endswith("_resemblyzer.npy"):
            user_id = file[len("voice_profile_"):-len("_resemblyzer.npy")]
            profiles[user_id] = np.load(file)
    if not profiles:
        speak("No voice profiles enrolled.")
        return None
    fs = 16000
    duration = 3
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    wav = np.squeeze(audio)
    wav = preprocess_wav(wav, source_sr=fs)
    embed = encoder.embed_utterance(wav)
    matched_user = None
    best_sim = 0
    for user_id, profile in profiles.items():
        sim = np.dot(embed, profile) / (np.linalg.norm(embed) * np.linalg.norm(profile))
        logger.info(f"Resemblyzer similarity with {user_id}: {sim:.2f}")
        if sim > 0.90 and sim > best_sim:
            matched_user = user_id
            best_sim = sim
    if matched_user:
        speak(f"Voice verified for user {matched_user}.")
        logger.info(f"Voice matched for {matched_user} (Resemblyzer).")
        return matched_user
    else:
        speak("Voice not recognized. Access denied.")
        logger.warning("No valid voice command detected or voice not recognized (Resemblyzer)")
        return None

def enroll_fingerprint():
    try:
        speak("Place your finger firmly on the sensor for enrollment")
        update_lcd_display("Fingerprint", "Place firmly")
        for attempt in range(3):
            logger.info(f"Enrollment attempt {attempt + 1}")
            timeout = time.time() + 15
            while not f.readImage() and time.time() < timeout:
                sleep(0.2)
            if time.time() >= timeout:
                speak("No finger detected. Try again.")
                update_lcd_display("No finger", "detected")
                continue
            try:
                f.convertImage(0x01)
                if f.downloadCharacteristics(0x01):
                    break
                else:
                    if attempt < 2:
                        speak("Poor image quality. Try again with firm pressure.")
                        update_lcd_display("Poor quality", "Press firmly")
                        sleep(2)
                        continue
                    else:
                        speak("Unable to capture clear fingerprint")
                        return False
            except Exception as conv_error:
                logger.warning(f"Conversion attempt {attempt + 1} failed: {str(conv_error)}")
                if attempt < 2:
                    speak("Image processing failed. Try again.")
                    sleep(1)
                    continue
                else:
                    speak("Fingerprint capture failed")
                    return False
        try:
            result = f.searchTemplate()
            if result[0] >= 0:
                speak("Fingerprint already registered")
                update_lcd_display("Already", "Registered")
                return False
        except Exception as search_error:
            logger.warning(f"Search template error: {str(search_error)}")
        speak("Remove finger and place again for confirmation")
        update_lcd_display("Remove finger", "Place again")
        sleep(3)
        for attempt in range(3):
            timeout = time.time() + 15
            while not f.readImage() and time.time() < timeout:
                sleep(0.2)
            if time.time() >= timeout:
                if attempt < 2:
                    speak("Place finger again")
                    continue
                else:
                    speak("Enrollment timeout")
                    return False
            try:
                f.convertImage(0x02)
                break
            except Exception as conv_error:
                logger.warning(f"Second conversion attempt {attempt + 1} failed: {str(conv_error)}")
                if attempt < 2:
                    speak("Try again with firm pressure")
                    sleep(1)
                    continue
                else:
                    speak("Second capture failed")
                    return False
        try:
            if f.compareCharacteristics() == 0:
                speak("Fingers do not match. Try enrollment again.")
                return False
        except Exception as compare_error:
            logger.error(f"Template comparison failed: {str(compare_error)}")
            speak("Template comparison failed")
            return False
        try:
            f.createTemplate()
            position = f.storeTemplate()
            speak(f"Fingerprint enrolled successfully at position {position}")
            update_lcd_display("Enrolled", f"Position: {position}")
            notification_manager.log_notification(user_id, "Fingerprint enrolled")
            return True
        except Exception as store_error:
            logger.error(f"Template storage failed: {str(store_error)}")
            speak("Failed to store fingerprint")
            return False
    except Exception as e:
        logger.error(f"Fingerprint enrollment error: {str(e)}")
        speak("Enrollment failed due to sensor error")
        return False

def verify_fingerprint():
    try:
        speak("hi boss Place finger on the sensor")
        update_lcd_display("Fingerprint", "Place firmly")
        timeout = time.time() + 12
        finger_detected = False
        while time.time() < timeout:
            try:
                if f.readImage():
                    finger_detected = True
                    break
            except Exception as read_error:
                logger.warning(f"Read image error: {str(read_error)}")
            sleep(0.2)
        if not finger_detected:
            speak("No finger detected")
            update_lcd_display("No finger", "detected")
            notification_manager.log_notification(user_id, "Fingerprint access denied - no finger")
            return False
        try:
            f.convertImage(0x01)
            if not f.downloadCharacteristics(0x01):
                speak("Poor image quality. Try again.")
                update_lcd_display("Poor quality", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - poor quality")
                return False
            result = f.searchTemplate()
            if result[0] >= 0:
                confidence = result[1]
                position = result[0]
                logger.info(f"Fingerprint matched at position {position} with confidence {confidence}")
                speak("Fingerprint verified successfully")
                update_lcd_display("Fingerprint", "Verified")
                notification_manager.log_notification(user_id, "Fingerprint access granted")
                auth_state.add_authentication("fingerprint")
                return True
            else:
                speak("Fingerprint not recognized")
                update_lcd_display("Not Recognized", "Access Denied")
                notification_manager.log_notification(user_id, "Fingerprint access denied - not recognized")
                return False
        except Exception as verify_error:
            error_msg = str(verify_error)
            logger.warning(f"Fingerprint verification failed: {error_msg}")
            if "too few feature points" in error_msg.lower():
                speak("Poor fingerprint quality. Clean finger and try again.")
                update_lcd_display("Clean finger", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - insufficient features")
            elif "timeout" in error_msg.lower():
                speak("Sensor timeout. Try again.")
                update_lcd_display("Sensor timeout", "Try again")
                notification_manager.log_notification(user_id, "Fingerprint access denied - timeout")
            else:
                speak("Fingerprint verification error")
                update_lcd_display("Verification", "Error")
                notification_manager.log_notification(user_id, "Fingerprint access denied - error")
            return False
    except Exception as e:
        logger.error(f"Fingerprint verification error: {str(e)}")
        speak("Verification failed due to sensor error")
        update_lcd_display("Sensor Error", "Try again")
        notification_manager.log_notification(user_id, "Fingerprint access denied - sensor error")
        return False

def read_keypad():
    for row_index, row_pin in enumerate(ROW_PINS):
        safe_gpio_operation(lambda: GPIO.output(row_pin, GPIO.HIGH))
        for col_index, col_pin in enumerate(COL_PINS):
            if GPIO.input(col_pin) == GPIO.HIGH:
                time.sleep(0.05)
                if GPIO.input(col_pin) == GPIO.HIGH:
                    key = KEYPAD[row_index][col_index]
                    timeout = time.time() + 2.0
                    while GPIO.input(col_pin) == GPIO.HIGH and time.time() < timeout:
                        time.sleep(0.01)
                    safe_gpio_operation(lambda: GPIO.output(row_pin, GPIO.LOW))
                    return key
        safe_gpio_operation(lambda: GPIO.output(row_pin, GPIO.LOW))
    return None

def speak(message):
    if system_mode.is_online:
        try:
            tts = gTTS(text=message, lang='en')
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                tts.save(fp.name)
                result = os.system(f"mpg321 {fp.name} > /dev/null 2>&1")
                if result == 0:
                    return
        except Exception as e:
            logger.warning(f"Online TTS failed: {str(e)}, switching to offline")
    try:
        temp_file = "/tmp/output.wav"
        command = f'pico2wave -w {temp_file} "{message}" && aplay {temp_file} > /dev/null 2>&1 && rm {temp_file}'
        result = os.system(command)
        if result == 0:
            return
        os.system(f'espeak "{message}" > /dev/null 2>&1')
    except Exception as e:
        logger.error(f"All TTS methods failed: {str(e)}")

def update_lcd_display(line1, line2=""):
    try:
        lcd.clear()
        lcd.write_string(line1[:16])
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:16])
    except Exception as e:
        logger.error(f"LCD update error: {str(e)}")

def sound_buzzer(duration=3):
    try:
        safe_gpio_operation(lambda: GPIO.output(PINS['buzzer'], GPIO.HIGH))
        sleep(duration)
        safe_gpio_operation(lambda: GPIO.output(PINS['buzzer'], GPIO.LOW))
    except Exception as e:
        logger.error(f"Buzzer error: {str(e)}")

def display_message(message, stop_event):
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

def is_button_pressed(pin, debounce_time=0.1):
    if GPIO.input(pin) == GPIO.LOW:
        time.sleep(debounce_time)
        if GPIO.input(pin) == GPIO.LOW:
            timeout = time.time() + 2.0
            while GPIO.input(pin) == GPIO.LOW and time.time() < timeout:
                time.sleep(0.01)
            return True
    return False

@app.route('/video_feed')
def video_feed():
    def generate():
        video_manager.start_streaming()
        try:
            while video_manager.is_streaming() and not video_manager.stop_event.is_set():
                frame = video_manager.get_frame()
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                time.sleep(0.125)  # Control frame rate to ~8 FPS
        except GeneratorExit:
            logger.info("Video feed client disconnected")
        except Exception as e:
            logger.error(f"Video feed error: {str(e)}")
        finally:
            video_manager.stop_streaming()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_solenoid', methods=['POST'])
def control_solenoid():
    try:
        switch_state = request.form.get("switch")
        logger.info(f"Solenoid control request: {switch_state}")
        def operation_callback(success):
            if success:
                logger.info(f"Solenoid {'activated' if switch_state == 'on' else 'deactivated'} successfully")
                notification_manager.log_notification(user_id, f"Solenoid {'activated' if switch_state == 'on' else 'deactivated'} via API")
            else:
                logger.error(f"Failed to {'activate' if switch_state == 'on' else 'deactivate'} solenoid")
        if switch_state == "on":
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), operation_callback))
            return jsonify({"status": "queued", "message": "Solenoid activation queued"}), 202
        elif switch_state == "off":
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), operation_callback))
            return jsonify({"status": "queued", "message": "Solenoid deactivation queued"}), 202
        else:
            return jsonify({"status": "error", "message": "Invalid switch state"}), 400
    except Exception as e:
        logger.error(f"Solenoid control error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_video')
def stop_video():
    video_manager.stop_streaming()
    return jsonify({"status": "success", "message": "Video streaming stopped"})

@app.route('/')
def index():
    return 'Smart Lock System with Two-Factor Authentication Running!'

def try_authentication(method, total_failures, max_failures):
    if method == "pin":
        speak("Enter your pin")
        update_lcd_display("Enter PIN:", "")
        a_pressed = True
        entered_password = ""
        pin_start_time = time.time()
        pin_timeout = 10.0
        while a_pressed:
            key = read_keypad()
            current_time = time.time()
            if key:
                logger.info(f"Key pressed: {key}")
                if key == "OK":
                    if entered_password == PASSWORD:
                        speak("PIN verified. Door unlocked.")
                        update_lcd_display("Access Granted!", "Door Unlocked")
                        def solenoid_callback(success):
                            if success:
                                logger.info("PIN solenoid unlock cycle completed")
                        solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
                        sleep(5)
                        solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
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
            if (current_time - pin_start_time) > pin_timeout:
                a_pressed = False
                entered_password = ""
                update_lcd_display("PIN Timeout", "Try again")
                speak("PIN entry timed out. Returning to normal verification.")
                total_failures += 1
                notification_manager.log_notification(user_id, "PIN access denied - timeout")
                sleep(2)
                return False, total_failures
            sleep(0.05)
        return False, total_failures
    elif method == "voice":
        speak("Verify your voice")
        update_lcd_display("Voice Verify", "Speak now")
        matched_user = listen_for_command_resemblyzer()
        if matched_user:
            speak(f"Voice verified for user {matched_user}. Door unlocked.")
            update_lcd_display("Access Granted!", f"User: {matched_user}")
            def solenoid_callback(success):
                if success:
                    logger.info("Voice solenoid unlock cycle completed")
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
            sleep(5)
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
            notification_manager.log_notification(matched_user, "Voice access granted (Resemblyzer)")
            return True, total_failures
        else:
            speak("Voice not recognized")
            update_lcd_display("Voice Failed", "Try again")
            notification_manager.log_notification(user_id, "Voice access denied (Resemblyzer)")
            return False, total_failures + 1
    elif method == "face":
        if verify_face():
            speak("Face verified. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            def solenoid_callback(success):
                if success:
                    logger.info("Face solenoid unlock cycle completed")
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
            sleep(5)
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
            return True, total_failures
        else:
            logger.info(f"Face verification failed. Total failures: {total_failures + 1}/{max_failures}")
            return False, total_failures + 1
    elif method == "fingerprint":
        if verify_fingerprint():
            speak("Fingerprint verified. Door unlocked.")
            update_lcd_display("Access Granted!", "Door Unlocked")
            def solenoid_callback(success):
                if success:
                    logger.info("Fingerprint solenoid unlock cycle completed")
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
            sleep(5)
            solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
            return True, total_failures
        else:
            logger.info(f"Fingerprint failed. Total failures: {total_failures + 1}/{max_failures}")
            return False, total_failures + 1
    return False, total_failures

def main_loop():
    total_failures = 0
    max_failures = 3
    update_lcd_display("Smart Lock Ready", "Select Method")
    system_mode.update_mode()
    mode_text = "Online Mode" if system_mode.is_online else "Offline Mode"
    update_lcd_display("Smart Lock Ready", mode_text)
    def on_mode_change(is_online):
        mode_text = "Online" if is_online else "Offline"
        update_lcd_display(f"{mode_text} Mode", "Active")
        notification_manager.log_notification(user_id, f"System switched to {mode_text} mode")
        if is_online:
            upload_offline_logs_and_images()
    system_mode.add_mode_change_callback(on_mode_change)
    fallback_sequence = []
    current_step = 0
    sequence_active = False
    pin_ready = False
    while True:
        try:
            system_mode.update_mode()
            if sequence_active and auth_state.unlock_complete:
                fallback_sequence = []
                current_step = 0
                sequence_active = False
                pin_ready = False
                auth_state.unlock_complete = False
                update_lcd_display("Smart Lock Ready", "Select Method")
                sleep(1)
            if not sequence_active:
                try:
                    if f.readImage():
                        logger.info("Fingerprint detected, starting sequence: fingerprint → voice → face")
                        fallback_sequence = ["fingerprint", "voice", "face"]
                        current_step = 0
                        sequence_active = True
                        pin_ready = False
                        continue
                except Exception as e:
                    logger.warning(f"Fingerprint sensor check failed: {str(e)}")
            key = read_keypad()
            if key:
                logger.info(f"Key pressed: {key}")
                if key == "A" and not pin_ready:
                    update_lcd_display("PIN Locked", "Try other methods")
                    speak("PIN is locked.")
                    notification_manager.log_notification(user_id, "PIN attempted while locked")
                    sleep(2)
                    continue
                if not sequence_active:
                    if key == "1":
                        logger.info("Voice selected, starting sequence: voice → fingerprint → face")
                        fallback_sequence = ["voice", "fingerprint", "face"]
                        current_step = 0
                        sequence_active = True
                        pin_ready = False
                        continue
                    elif key == "2":
                        logger.info("Face selected, starting sequence: face → fingerprint → voice")
                        fallback_sequence = ["face", "fingerprint", "voice"]
                        current_step = 0
                        sequence_active = True
                        pin_ready = False
                        continue
                else:
                    if key == "A":
                        if pin_ready:
                            success, total_failures = try_authentication("pin", total_failures, max_failures)
                            fallback_sequence = []
                            current_step = 0
                            sequence_active = False
                            pin_ready = False
                            auth_state.unlock_complete = False
                            update_lcd_display("Smart Lock Ready", "Select Method")
                            sleep(1)
                        continue
                    elif key == "OK":
                        update_lcd_display("Invalid Input", "Try again")
                        sleep(2)
                        continue
            if sequence_active and current_step < len(fallback_sequence):
                method = fallback_sequence[current_step]
                success, total_failures = try_authentication(method, total_failures, max_failures)
                current_step += 1
                if success:
                    total_failures = 0
                    sleep(2)
                    fallback_sequence = []
                    current_step = 0
                    sequence_active = False
                    pin_ready = False
                    auth_state.unlock_complete = False
                    update_lcd_display("Smart Lock Ready", "Select Method")
                    sleep(1)
                elif total_failures >= max_failures:
                    save_intruder_image()
                    sound_buzzer(8)
                    total_failures = 0
                    pin_ready = True
                    pin_window_start = time.time()
                    update_lcd_display("PIN Available")
                    speak("You may now enter your PIN")
                    pin_activated = False
                    while time.time() - pin_window_start < 10:
                        key = read_keypad()
                        if key == "A":
                            pin_activated = True
                            success, total_failures = try_authentication("pin", total_failures, max_failures)
                            fallback_sequence = []
                            current_step = 0
                            sequence_active = False
                            pin_ready = False
                            auth_state.unlock_complete = False
                            update_lcd_display("Smart Lock Ready", "Select Method")
                            sleep(1)
                            break
                        sleep(0.05)
                    if not pin_activated:
                        pin_ready = False
                        fallback_sequence = []
                        current_step = 0
                        sequence_active = False
                        auth_state.unlock_complete = False
                        update_lcd_display("Smart Lock Ready", "Select Method")
                        speak("PIN entry expired.")
                        sleep(1)
            if sequence_active and current_step >= len(fallback_sequence):
                pin_ready = True
                update_lcd_display("PIN Available.")
                speak("You may now enter your PIN.")
            if is_button_pressed(PINS['button1']):
                speak("Register your voice profile")
                update_lcd_display("Voice Register", "Speak now")
                success = enroll_voice_resemblyzer()
                if success:
                    speak("Voice profile enrolled successfully")
                    update_lcd_display("Voice", "Enrolled")
                    notification_manager.log_notification(user_id, "Voice profile enrolled (Resemblyzer)")
                else:
                    speak("Voice enrollment failed")
                    update_lcd_display("Voice", "Enrollment Failed")
                sleep(2)
            if is_button_pressed(PINS['button5']):
                enroll_fingerprint()
                sleep(2)
            if is_button_pressed(PINS['button6']):
                speak("Door open")
                update_lcd_display("Door Opened")
                def solenoid_callback(success):
                    if success:
                        logger.info("Manual solenoid unlock cycle completed")
                solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.HIGH), None))
                sleep(5)
                solenoid_queue.put((lambda: GPIO.output(PINS['solenoid'], GPIO.LOW), solenoid_callback))
                notification_manager.log_notification(user_id, "Manual unlock used")
                sleep(1)
            if is_button_pressed(PINS['button7']):
                capture_and_save_face()
                sleep(2)
            sleep(0.05)
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            sleep(0.5)

def upload_offline_logs_and_images():
    """Upload offline logs and images to the database when online, skipping duplicates."""
    # Ensure database connection
    db_manager.ensure_connection()
    if not (system_mode.is_online and db_manager.is_connected()):
        logger.warning("Cannot upload offline data: system offline or database not connected")
        return

    # Upload offline logs
    offline_log_file = notification_manager.offline_log_file
    if os.path.exists(offline_log_file):
        try:
            with open(offline_log_file, "r") as log_file:
                lines = log_file.readlines()
            temp_file = f"{offline_log_file}.tmp"
            with open(temp_file, "w") as temp_log_file:
                for line in lines:
                    try:
                        parts = line.strip().split(" - ")
                        if len(parts) >= 3:
                            timestamp = parts[0].split("]")[0].replace("[ONLINE", "").replace("[OFFLINE", "").strip("[] ")
                            user_part = parts[1].replace("User: ", "").strip()
                            event_part = parts[2].replace("Event: ", "").strip()
                            # Check for duplicates
                            query = "SELECT COUNT(*) as count FROM logs WHERE user_id=%s AND notify=%s AND timestamp=%s"
                            db_manager.cursor.execute(query, (user_part, event_part, timestamp))
                            result = db_manager.cursor.fetchone()
                            if result and result["count"] == 0:
                                insert_query = "INSERT INTO logs (user_id, notify, timestamp) VALUES (%s, %s, %s)"
                                db_manager.cursor.execute(insert_query, (user_part, event_part, timestamp))
                                db_manager.connection.commit()
                                logger.info(f"Uploaded offline log: {line.strip()}")
                            else:
                                logger.info(f"Skipped duplicate log: {line.strip()}")
                        else:
                            logger.warning(f"Invalid log format, skipping: {line.strip()}")
                            temp_log_file.write(line)  # Keep invalid lines for retry
                    except Exception as e:
                        logger.warning(f"Failed to upload offline log '{line.strip()}': {str(e)}")
                        temp_log_file.write(line)  # Keep failed lines for retry
            # Replace original file with temp file (only failed or invalid logs remain)
            os.replace(temp_file, offline_log_file)
        except Exception as e:
            logger.error(f"Error processing offline logs: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Upload offline images
    offline_dir = "offline_images"
    if os.path.exists(offline_dir):
        for filename in os.listdir(offline_dir):
            if filename.endswith(".jpg"):
                image_path = os.path.join(offline_dir, filename)
                try:
                    with open(image_path, 'rb') as file:
                        binary_data = file.read()
                    import hashlib
                    img_hash = hashlib.sha256(binary_data).hexdigest()
                    query = "SELECT COUNT(*) as count FROM images WHERE SHA2(image, 256)=%s"
                    db_manager.cursor.execute(query, (img_hash,))
                    result = db_manager.cursor.fetchone()
                    if result and result["count"] == 0:
                        insert_query = "INSERT INTO images (image) VALUES (%s)"
                        db_manager.cursor.execute(insert_query, (binary_data,))
                        db_manager.connection.commit()
                        logger.info(f"Uploaded offline image: {filename}")
                        os.remove(image_path)
                    else:
                        logger.info(f"Skipped duplicate image: {filename}")
                        os.remove(image_path)
                except Exception as e:
                    logger.warning(f"Failed to upload offline image {filename}: {str(e)}")

def get_frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def adapt_camera_to_brightness():
    frame = video_manager.get_frame()
    if frame is None:
        return
    brightness = get_frame_brightness(frame)
    if brightness < 60:
        picam2.set_controls({"AnalogueGain": 16.0, "ExposureTime": 1000000})
    elif brightness < 100:
        picam2.set_controls({"AnalogueGain": 8.0, "ExposureTime": 500000})
    elif brightness < 160:
        picam2.set_controls({"AnalogueGain": 2.0, "ExposureTime": 100000})
    else:
        picam2.set_controls({"AnalogueGain": 1.0, "ExposureTime": 50000})

# UDP Audio Streaming Configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SIZE = 16000
AUDIO_BUFFER = queue.Queue(maxsize=10)

# Initialize UDP socket for audio
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)
logging.info(f"Listening for UDP audio on {UDP_IP}:{UDP_PORT}")

def audio_playback_thread():
    stream = None
    try:
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, blocksize=BUFFER_SIZE)
        stream.start()
        logging.debug("Playback stream started")
        while True:
            try:
                audio_data = AUDIO_BUFFER.get(timeout=1.0)
                if audio_data is None:
                    logging.debug("Received None, stopping playback")
                    break
                if stream.active:
                    logging.debug(f"Playing audio chunk of size {len(audio_data)} bytes")
                    stream.write(audio_data)
                else:
                    logging.warning("Stream not active, attempting to restart")
                    stream.close()
                    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, blocksize=BUFFER_SIZE)
                    stream.start()
                    stream.write(audio_data)
            except queue.Empty:
                logging.debug("Audio buffer empty, waiting")
                continue
            except Exception as e:
                logging.error(f"Playback thread error: {str(e)}")
                if stream and not stream.active:
                    logging.warning("Stream became inactive, attempting to restart")
                    stream.close()
                    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, blocksize=BUFFER_SIZE)
                    stream.start()
    except Exception as e:
        logging.error(f"Playback thread initialization error: {str(e)}")
    finally:
        if stream:
            stream.stop()
            stream.close()
            logging.info("Playback stream closed")

# Start playback thread for UDP audio
playback_thread = threading.Thread(target=audio_playback_thread, daemon=True)
playback_thread.start()

def udp_audio_receiver_thread():
    try:
        while True:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE * 2)
                logging.info(f"Received {len(data)} bytes from {addr}")
                if len(data) < 1:
                    logging.warning("Received empty packet")
                    continue
                if len(data) % 2 != 0:
                    logging.warning(f"Invalid data length {len(data)}, truncating to even")
                    data = data[:len(data) - 1]
                try:
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    if audio_array.size == 0:
                        logging.warning("Empty audio array after conversion")
                        continue
                    logging.debug(f"Audio array shape: {audio_array.shape}, min: {audio_array.min()}, max: {audio_array.max()}")
                    try:
                        AUDIO_BUFFER.put(audio_array, block=False)
                        logging.debug("Added audio to buffer")
                    except queue.Full:
                        logging.warning("Audio buffer full, dropping packet")
                except Exception as e:
                    logging.error(f"Error converting audio data: {e}")
            except socket.timeout:
                logging.debug("Socket timeout, continuing")
                continue
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
    except KeyboardInterrupt:
        logging.info("UDP audio receiver stopped by user")
    finally:
        AUDIO_BUFFER.put(None)
        sock.close()
        logging.info("UDP socket closed")

# Start UDP audio receiver thread
udp_receiver_thread = threading.Thread(target=udp_audio_receiver_thread, daemon=True)
udp_receiver_thread.start()

if __name__ == '__main__':
    try:
        check_offline_dependencies()
        logger.info("Smart Lock System Starting...")
        speak("System ready.")
        solenoid_worker_thread = threading.Thread(target=solenoid_worker, daemon=True)
        solenoid_worker_thread.start()
        time.sleep(0.2)
        def run_flask():
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
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
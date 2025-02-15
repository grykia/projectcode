import dlib
import numpy as np
import cv2
import sqlite3
import logging
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from mfrc522 import SimpleMFRC522
import RPi.GPIO as GPIO
import hashlib
import time
import signal
import sys

# Pin Configuration
YELLOW_LED_PIN = 17  # GPIO17
RED_LED_PIN = 27     # GPIO27
BUZZER_PIN = 22      # GPIO22

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize GPIO mode
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Setup GPIO pins
GPIO.setup(YELLOW_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

class IntegratedAttendanceSystem:
    def __init__(self):
        try:
            # Initialize hardware
            self.init_hardware()
            
            # Initialize RFID reader
            self.reader = SimpleMFRC522()
            
            # Initialize Firebase
            if not firebase_admin._apps:
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            
            # Initialize local database
            self.init_local_db()
            
            # Face recognition setup
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
            self.face_reco_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
            
            # Session and temporary attendance management
            self.current_session = None
            self.temp_attendance = {}
            self.lecturer_rfid = None
            self.persistent_rfid_attendance = {}
            self.new_rfid_attendance = {}
            
            logging.info("Integrated Attendance System initialized successfully")
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            raise

    def init_hardware(self):
        """Initialize LED and buzzer pins"""
        GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    def single_beep(self, duration=0.1):
        """Generate a single beep"""
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

    def multiple_beeps(self, count=3, duration=0.1, interval=0.1):
        """Generate multiple beeps"""
        for _ in range(count):
            self.single_beep(duration)
            time.sleep(interval)

    def blink_led(self, led_pin, duration=0.5):
        """Blink specified LED"""
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(led_pin, GPIO.LOW)

    def both_leds_on(self):
        """Turn on both LEDs"""
        GPIO.output(YELLOW_LED_PIN, GPIO.HIGH)
        GPIO.output(RED_LED_PIN, GPIO.HIGH)

    def both_leds_off(self):
        """Turn off both LEDs"""
        GPIO.output(YELLOW_LED_PIN, GPIO.LOW)
        GPIO.output(RED_LED_PIN, GPIO.LOW)

    def init_local_db(self):
        """Initialize the local SQLite database"""
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS students
                          (unique_id TEXT PRIMARY KEY, 
                           name TEXT NOT NULL, 
                           rfid_id TEXT UNIQUE,
                           face_descriptors BLOB)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS lecturers
                          (unique_id TEXT PRIMARY KEY, 
                           name TEXT NOT NULL, 
                           rfid_id TEXT UNIQUE,
                           course_name TEXT NOT NULL,
                           course_code TEXT NOT NULL)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS attendance_log
                          (unique_id TEXT NOT NULL,
                           name TEXT NOT NULL,
                           date TEXT NOT NULL,
                           session_id TEXT NOT NULL,
                           rfid_time TEXT,
                           face_time TEXT,
                           status TEXT)''')
            
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise

    def generate_unique_id(self, name):
        """Generate a consistent unique ID for a student or lecturer"""
        return hashlib.sha256(name.encode()).hexdigest()[:16]

    def is_rfid_registered(self, rfid_id):
        """Check if an RFID is already registered"""
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT rfid_id FROM students WHERE rfid_id=?", (str(rfid_id),))
        student_rfid = cursor.fetchone()
        
        cursor.execute("SELECT rfid_id FROM lecturers WHERE rfid_id=?", (str(rfid_id),))
        lecturer_rfid = cursor.fetchone()
        
        conn.close()
        
        return student_rfid is not None or lecturer_rfid is not None

    def capture_face_sample(self, name, timeout=30):
        """Capture a single face sample with robust error handling"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Capturing face for {name}. Look directly at the camera. Press 'q' to exit.")
        
        start_time = time.time()
        face_descriptor = None
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                time.sleep(0.1)
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            
            if faces:
                face = faces[0]
                shape = self.predictor(gray, face)
                face_descriptor = self.face_reco_model.compute_face_descriptor(frame, shape)
                face_descriptor = np.array(face_descriptor)
                
                cv2.rectangle(frame, 
                    (face.left(), face.top()), 
                    (face.right(), face.bottom()), 
                    (0, 255, 0), 2)
                cv2.imshow('Face Registration', frame)
                cv2.waitKey(500)
                
                break
            
            cv2.imshow('Face Registration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return face_descriptor

    def capture_face_samples(self, name, unique_id):
        """Capture multiple face samples"""
        face_descriptors = []
        
        prompts = [
            f"{name}, look directly at the camera",
            f"{name}, turn slightly to the left",
            f"{name}, turn slightly to the right"
        ]
        
        for prompt in prompts:
            input(f"\nPress Enter to capture: {prompt}")
            
            face_descriptor = self.capture_face_sample(name)
            if face_descriptor is not None:
                face_descriptors.append(face_descriptor)
            else:
                print(f"Failed to capture face for: {prompt}")
        
        return face_descriptors

    def register_student(self):
        """Register a new student"""
        try:
            print("Place RFID card on the reader...")
            rfid_id, _ = self.reader.read()
            
            if self.is_rfid_registered(rfid_id):
                print("Error: RFID card already registered.")
                self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
                return None
            
            name = input("Enter student name: ")
            unique_id = self.generate_unique_id(name)
            
            print("Prepare for face registration. Follow on-screen instructions.")
            face_descriptors = self.capture_face_samples(name, unique_id)
            
            if not face_descriptors or len(face_descriptors) < 2:
                print("Face registration incomplete. Please try again.")
                self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
                return None
            
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            serialized_descriptors = np.array(face_descriptors).tobytes()
            
            try:
                cursor.execute("""
                    INSERT INTO students 
                    (unique_id, name, rfid_id, face_descriptors) 
                    VALUES (?, ?, ?, ?)
                """, (unique_id, name, str(rfid_id), serialized_descriptors))
                
                conn.commit()
                print(f"Successfully registered student: {name}")
                self.blink_led(YELLOW_LED_PIN, 1.0)  # Success feedback
                self.single_beep(0.2)
                return unique_id
            
            except sqlite3.IntegrityError:
                print("Error: RFID card already registered to another student.")
                self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
                return None
            
            finally:
                conn.close()
            
        except Exception as e:
            logging.error(f"Student registration error: {e}")
            self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
            return None

    def register_lecturer(self):
        """Register a new lecturer"""
        try:
            print("Place lecturer RFID card on the reader...")
            rfid_id, _ = self.reader.read()
            
            if self.is_rfid_registered(rfid_id):
                print("Error: RFID card already registered.")
                self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
                return None
            
            name = input("Enter lecturer name: ")
            course_name = input("Enter course name: ")
            course_code = input("Enter course code: ")
            unique_id = self.generate_unique_id(name)
            
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO lecturers 
                    (unique_id, name, rfid_id, course_name, course_code) 
                    VALUES (?, ?, ?, ?, ?)
                """, (unique_id, name, str(rfid_id), course_name, course_code))
                
                conn.commit()
                print(f"Successfully registered lecturer: {name}")
                self.blink_led(YELLOW_LED_PIN, 1.0)  # Success feedback
                self.multiple_beeps(2, 0.2, 0.1)  # Double beep for lecturer
                return unique_id
            
            except sqlite3.IntegrityError:
                print("Error: RFID card already registered to another lecturer.")
                self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
                return None
            
            finally:
                conn.close()
            
        except Exception as e:
            logging.error(f"Lecturer registration error: {e}")
            self.blink_led(RED_LED_PIN, 1.0)  # Error feedback
            return None

    def mark_attendance_rfid(self):
        """Mark attendance using RFID with persistent records and LED/buzzer feedback"""
        try:
            print("Starting continuous attendance process...")
            print("Place RFID cards on the reader. System will verify faces when lecturer taps their card.")
            
            while True:
                rfid_id, _ = self.reader.read()
                
                conn = sqlite3.connect("attendance.db")
                cursor = conn.cursor()
                
                cursor.execute("SELECT unique_id, name, course_name, course_code FROM lecturers WHERE rfid_id=?", (str(rfid_id),))
                lecturer = cursor.fetchone()
                
                if lecturer:
                    print("\nLecturer card detected. Starting new session...")
                    lecturer_id, lecturer_name, course_name, course_code = lecturer
                    conn.close()
                    
                    # Lecturer feedback
                    self.both_leds_on()
                    self.multiple_beeps(3, 0.1, 0.1)
                    
                    session_id = f"{course_code}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    session_ref = self.db.collection('lectures').document(session_id)
                    session_ref.set({
                        'course_name': course_name,
                        'course_code': course_code,
                        'lecturer_name': lecturer_name,
                        'lecturer_rfid': str(rfid_id),
                        'start_time': datetime.datetime.now().isoformat(),
                        'status': 'active'
                    })
                    
                    self.current_session = session_id
                    print(f"New session created: {session_id}")
                    
                    self.persistent_rfid_attendance.update(self.new_rfid_attendance)
                    
                    print("\nStarting face verification process...")
                    self.verify_attendance(self.persistent_rfid_attendance)
                    
                    self.new_rfid_attendance = {}
                    self.both_leds_off()
                    
                    print("\nStarting new RFID attendance collection...")
                    continue
                
                cursor.execute("SELECT unique_id, name FROM students WHERE rfid_id=?", (str(rfid_id),))
                student = cursor.fetchone()
                
                if not student:
                    print("Unregistered RFID card")
                    self.blink_led(RED_LED_PIN, 0.5)
                    conn.close()
                    continue
                
                unique_id, name = student
                
                if unique_id in self.persistent_rfid_attendance:
                    print(f"RFID already registered for {name}. No need to tap again.")
                    self.blink_led(RED_LED_PIN, 0.5)  # Red LED for duplicate
                    conn.close()
                    continue
                
                if unique_id in self.new_rfid_attendance:
                    print(f"RFID already registered for {name} in current session.")
                    self.blink_led(RED_LED_PIN, 0.5)  # Red LED for duplicate
                    conn.close()
                    continue
                
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                self.new_rfid_attendance[unique_id] = {
                    'name': name,
                    'rfid_time': current_time
                }
                
                # Success feedback for new attendance
                self.blink_led(YELLOW_LED_PIN, 0.5)
                self.single_beep(0.1)
                
                print(f"New RFID registration for {name} at {current_time}")
                print("Waiting for more new students or lecturer to start session verification...")
                conn.close()
                
        except Exception as e:
            logging.error(f"RFID attendance error: {e}")
            print(f"Error: {e}")
            self.both_leds_off()  # Ensure LEDs are off in case of error

    def verify_attendance(self, attendance_dict):
        """Mark final attendance using face recognition with LED/buzzer feedback"""
        try:
            if not self.current_session:
                print("No active session. Create a session first.")
                return
            
            # Keep LEDs on during verification
            self.both_leds_on()
            
            for unique_id, data in attendance_dict.items():
                name = data['name']
                rfid_time = data['rfid_time']
                
                conn = sqlite3.connect("attendance.db")
                cursor = conn.cursor()
                cursor.execute("SELECT face_descriptors FROM students WHERE unique_id=?", (unique_id,))
                stored_descriptors = cursor.fetchone()
                
                if not stored_descriptors:
                    print(f"No face descriptors found for {name}")
                    continue
                
                face_descriptors = np.frombuffer(stored_descriptors[0]).reshape(-1, 128)
                
                print(f"\nVerifying face for {name}...")
                face_verified = self.verify_face(face_descriptors)
                
                current_time = datetime.datetime.now()
                current_date = current_time.strftime('%Y-%m-%d')
                status = 'present' if face_verified else 'partial'
                
                cursor.execute("""
                    INSERT INTO attendance_log 
                    (unique_id, name, date, session_id, rfid_time, face_time, status) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (unique_id, name, current_date, self.current_session, 
                      rfid_time, 
                      current_time.strftime('%H:%M:%S') if face_verified else None, 
                      status))
                
                conn.commit()
                conn.close()
                
                # Log to Firebase
                self.log_attendance_to_firebase(unique_id, name, status)
                
                # Feedback for face verification result
                if face_verified:
                    self.blink_led(YELLOW_LED_PIN, 0.5)
                    self.single_beep(0.1)
                else:
                    self.blink_led(RED_LED_PIN, 0.5)
                
                print(f"Final attendance marked for {name}: {status}")
            
        except Exception as e:
            logging.error(f"Face recognition attendance error: {e}")
            print(f"Error: {e}")
        finally:
            self.both_leds_off()

    def verify_face(self, stored_descriptors, threshold=0.6):
        """Verify face using stored descriptors"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        face_verified = False
        start_time = time.time()
        exit_flag = False
        
        while (time.time() - start_time) < 30 and not exit_flag:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                time.sleep(0.1)
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            
            for face in faces:
                shape = self.predictor(gray, face)
                face_descriptor = self.face_reco_model.compute_face_descriptor(frame, shape)
                
                for stored_descriptor in stored_descriptors:
                    distance = np.linalg.norm(stored_descriptor - face_descriptor)
                    if distance < threshold:
                        face_verified = True
                        break
                
                if face_verified:
                    break
            
            cv2.imshow('Face Verification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_flag = True
        
        cap.release()
        cv2.destroyAllWindows()
        
        return face_verified

    def log_attendance_to_firebase(self, unique_id, name, status):
        """Log attendance to Firebase"""
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute("SELECT rfid_id FROM students WHERE unique_id=?", (unique_id,))
            rfid_result = cursor.fetchone()
            conn.close()
            
            if not rfid_result:
                logging.error(f"RFID not found for student {name}")
                return
                
            rfid_id = rfid_result[0]
            current_time = datetime.datetime.now()
            session_ref = self.db.collection('lectures').document(self.current_session)
            
            attendance_ref = session_ref.collection('attendance').document(str(rfid_id))
            attendance_ref.set({
                'unique_id': unique_id,
                'rfid_id': str(rfid_id),
                'name': name,
                'date': current_time.strftime('%Y-%m-%d'),
                'time': current_time.strftime('%H:%M:%S'),
                'status': status
            })
        except Exception as e:
            logging.error(f"Firebase logging error: {e}")

    def cleanup(self):
        """Cleanup GPIO pins"""
        self.both_leds_off()
        GPIO.cleanup()

    def run(self):
        """Main menu system"""
        try:
            while True:
                print("\nIntegrated Attendance System")
                print("1. Register New Student")
                print("2. Register Lecturer")
                print("3. Start Continuous Attendance Process")
                print("4. Exit")
                
                choice = input("Select option: ")
                
                if choice == '1':
                    self.register_student()
                elif choice == '2':
                    self.register_lecturer()
                elif choice == '3':
                    try:
                        self.mark_attendance_rfid()
                    except KeyboardInterrupt:
                        print("\nContinuous attendance process stopped.")
                        self.temp_attendance = {}
                        self.current_session = None
                        self.both_leds_off()
                elif choice == '4':
                    break
                else:
                    print("Invalid option")
        finally:
            self.cleanup()

def main():
    try:
        system = IntegratedAttendanceSystem()
        system.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
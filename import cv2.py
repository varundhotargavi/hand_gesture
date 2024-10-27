import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import threading
import time

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Get the screen size for scaling cursor movement
screen_width, screen_height = pyautogui.size()

# Initialize the video capture object with a higher resolution
cap = cv2.VideoCapture(0)  # 0 means the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Variables for gesture recognition and click simulation
prev_gesture = None
click_threshold = 15  # Threshold distance for click gesture
dragging = False
drag_start = None

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Function to recognize speech commands
def recognize_speech():
    while True:
        with sr.Microphone() as source:
            print("Listening for voice commands...")
            audio = recognizer.listen(source, phrase_time_limit=5)
            try:
                command = recognizer.recognize_google(audio)
                print("You said:", command)
                process_voice_command(command)
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
            time.sleep(1)

# Function to process voice commands
def process_voice_command(command):
    if "open" in command and "browser" in command:
        pyautogui.hotkey('ctrl', 't')
    elif "close" in command and "window" in command:
        pyautogui.hotkey('alt', 'f4')
    # Add more voice commands as needed

# Function to process each frame for hand detection
def process_frame(frame):
    global prev_gesture, dragging, drag_start
    
    # Apply Gaussian blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform hand detection
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and move cursor
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Get the tip of the thumb (landmark 4)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Convert the normalized coordinates to pixel values
            frame_height, frame_width, _ = frame.shape
            x = int(index_finger_tip.x * frame_width)
            y = int(index_finger_tip.y * frame_height)
            
            # Scale the coordinates to the screen size
            screen_x = np.interp(x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(y, [0, frame_height], [0, screen_height])
            
            # Move the cursor
            pyautogui.moveTo(screen_x, screen_y)
            
            # Detect click gesture (left-click)
            if prev_gesture is not None:
                dist = np.sqrt((screen_x - prev_gesture[0]) ** 2 + (screen_y - prev_gesture[1]) ** 2)
                if dist < click_threshold:
                    pyautogui.click()
                    prev_gesture = None
            
            # Detect right-click gesture (index finger and thumb pinch)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)
            thumb_screen_x = np.interp(thumb_x, [0, frame_width], [0, screen_width])
            thumb_screen_y = np.interp(thumb_y, [0, frame_height], [0, screen_height])
            
            thumb_dist = np.sqrt((screen_x - thumb_screen_x) ** 2 + (screen_y - thumb_screen_y) ** 2)
            if thumb_dist < click_threshold:
                pyautogui.rightClick()
                prev_gesture = None
            
            # Detect drag-and-drop gesture (index finger and thumb pinch and drag)
            if not dragging and thumb_dist < click_threshold:
                dragging = True
                drag_start = (screen_x, screen_y)
            elif dragging and thumb_dist >= click_threshold:
                pyautogui.dragTo(screen_x, screen_y, button='left')
                dragging = False
            
            # Update previous gesture position
            prev_gesture = (screen_x, screen_y)
    
    return frame

# Start the speech recognition in a separate thread
speech_thread = threading.Thread(target=recognize_speech)
speech_thread.daemon = True
speech_thread.start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Process the frame for hand detection and cursor movement
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Hand Controlled Cursor', processed_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

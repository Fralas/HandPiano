import cv2
import mediapipe as mp
import os
from threading import Thread, Lock
import pygame  # For audio playback
import time  # For debounce logic
import tkinter as tk
from tkinter import ttk


# Initialize pygame mixer
pygame.mixer.init()

# Function to load notes from the specified voice folder
def load_notes(voice_folder):
    audio_dir = os.path.join(os.path.dirname(__file__), "../audio", voice_folder)
    return {
        "A": pygame.mixer.Sound(os.path.join(audio_dir, "A3.mp3")),
        "B": pygame.mixer.Sound(os.path.join(audio_dir, "B3.mp3")),
        "C": pygame.mixer.Sound(os.path.join(audio_dir, "C3.mp3")),
        "D": pygame.mixer.Sound(os.path.join(audio_dir, "D3.mp3")),
        "E": pygame.mixer.Sound(os.path.join(audio_dir, "E3.mp3")),
        "F": pygame.mixer.Sound(os.path.join(audio_dir, "F3.mp3")),
    }

notes = None  # Notes will be loaded based on user selection

# Lock to prevent multiple threads from playing simultaneously
note_lock = Lock()

# Function to play a note in a separate thread (non-blocking playback)
def play_note_in_thread(note):
    try:
        with note_lock:
            if note in notes:
                print(f"Playing note: {note}")
                notes[note].play()
    except Exception as e:
        print(f"Error playing note: {e}")

# Function to get available cameras
def get_camera_indexes(max_cameras=5):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Function to handle camera and note-playing logic
def open_camera_with_hand_tracking(camera_index, voice_folder):
    global notes
    notes = load_notes(voice_folder)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    # Variables for debounce logic
    last_note = None
    last_note_time = 0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Distinguish between left and right hands
                if hand_label.classification[0].label == 'Left':  # Left hand
                    is_left_hand = True
                else:  # Right hand
                    is_left_hand = False

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                h, w, _ = frame.shape
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
                ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
                pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Logic for left hand (existing notes A, B, C, D)
                if is_left_hand:
                    if abs(index_x - thumb_x) < 20 and abs(index_y - thumb_y) < 20:
                        detected_note = "A"
                    elif abs(middle_x - thumb_x) < 20 and abs(middle_y - thumb_y) < 20:
                        detected_note = "B"
                    elif abs(ring_x - thumb_x) < 20 and abs(ring_y - thumb_y) < 20:
                        detected_note = "C"
                    elif abs(pinky_x - thumb_x) < 20 and abs(pinky_y - thumb_y) < 20:
                        detected_note = "D"
                    else:
                        detected_note = None

                # Logic for right hand (new notes E, F)
                else:
                    if abs(index_x - thumb_x) < 20 and abs(index_y - thumb_y) < 20:
                        detected_note = "E"  # Thumb and index of right hand
                    elif abs(middle_x - thumb_x) < 20 and abs(middle_y - thumb_y) < 20:
                        detected_note = "F"  # Thumb and middle of right hand
                    else:
                        detected_note = None

                # Debounce logic to prevent note spamming
                if detected_note:
                    current_time = time.time()
                    if detected_note != last_note or (current_time - last_note_time) > 0.5:
                        last_note = detected_note
                        last_note_time = current_time

                        # Play the note in a separate thread to avoid blocking
                        play_thread = Thread(target=play_note_in_thread, args=(detected_note,))
                        play_thread.start()

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(f"Hand Tracking Piano - Camera {camera_index}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter GUI for camera and voice selection
def start_camera():
    camera_index = int(camera_dropdown_var.get().split(' ')[1])
    voice_folder = voice_dropdown_var.get()
    open_camera_with_hand_tracking(camera_index, voice_folder)

root = tk.Tk()
root.title("Hand Tracking Piano")

# Dropdown menu to select camera
available_cameras = get_camera_indexes()
camera_dropdown_var = tk.StringVar(root)
camera_dropdown_var.set("Camera 0")
camera_options = [f"Camera {i}" for i in available_cameras]
camera_dropdown = ttk.Combobox(root, textvariable=camera_dropdown_var, values=camera_options)
camera_dropdown.pack(pady=10)

# Dropdown menu to select voice folder
voice_dropdown_var = tk.StringVar(root)
voice_dropdown_var.set("piano_voice")
voice_options = ["piano_voice", "emma_voice", "drumkit_voice"]
voice_dropdown = ttk.Combobox(root, textvariable=voice_dropdown_var, values=voice_options)
voice_dropdown.pack(pady=10)

# Start button
start_button = tk.Button(root, text="Start", command=start_camera)
start_button.pack(pady=20)

root.mainloop()

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from pydub import AudioSegment
from threading import Thread, Lock
import pygame  # For audio playback

# Initialize pygame mixer
pygame.mixer.init()

# Load the notes into pygame Sound objects for faster playback
def load_notes():
    audio_dir = os.path.join(os.path.dirname(__file__), "../audio")
    return {
        "C": pygame.mixer.Sound(os.path.join(audio_dir, "C3.mp3")),
        "D": pygame.mixer.Sound(os.path.join(audio_dir, "D3.mp3")),
        "E": pygame.mixer.Sound(os.path.join(audio_dir, "E3.mp3")),
        "F": pygame.mixer.Sound(os.path.join(audio_dir, "F3.mp3")),
        "G": pygame.mixer.Sound(os.path.join(audio_dir, "G3.mp3")),
        "A": pygame.mixer.Sound(os.path.join(audio_dir, "A3.mp3")),
        "B": pygame.mixer.Sound(os.path.join(audio_dir, "B3.mp3")),
    }

notes = load_notes()

# Lock to prevent multiple threads from playing simultaneously
note_lock = Lock()

# Flag to track if a note is being played
note_playing = False

# Function to play a note in a separate thread (non-blocking playback)
def play_note_in_thread(note):
    global note_playing
    try:
        with note_lock:
            if note in notes and not note_playing:
                print(f"Playing note: {note}")
                note_playing = True
                # Play the note using pygame Sound
                notes[note].play()

                # Set note_playing to False when the note finishes playing
                # pygame.mixer.Sound doesn't block, so no need to wait
                pygame.mixer.Sound.play(notes[note])
                note_playing = False
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

# Function to draw a piano on the canvas
def draw_piano_on_canvas(canvas, min_x, min_y, max_x, max_y):
    h, w, _ = canvas.shape
    piano_width = max_x - min_x
    piano_height = max_y - min_y
    num_keys = 7  # Fixed 7 keys (C, D, E, F, G, A, B)
    key_width = piano_width // num_keys
    key_height = piano_height

    for i in range(num_keys):
        x_start = min_x + i * key_width
        cv2.rectangle(canvas, (x_start, min_y), (x_start + key_width, min_y + key_height), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x_start, min_y), (x_start + key_width, min_y + key_height), (0, 0, 0), 2)
        note = "CDEFGAB"[i % 7]
        cv2.putText(canvas, note, (x_start + 10, min_y + key_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i in range(1, num_keys - 1):
        if i % 7 in [2, 6]:
            continue
        x_start = min_x + i * key_width + key_width // 2 - key_width // 4
        black_key_width = key_width // 2
        black_key_height = int(key_height // 1.5)
        cv2.rectangle(canvas, (x_start, min_y), (x_start + black_key_width, min_y + black_key_height), (0, 0, 0), -1)

# Function to map finger position to notes
def detect_note_from_position(x, min_x, max_x, num_keys=7):
    key_width = (max_x - min_x) // num_keys
    key_index = (x - min_x) // key_width
    note = "CDEFGAB"[key_index % 7]
    return note

# Function to handle camera and piano logic
def open_camera_with_piano(camera_index):
    global note_playing
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    canvas = None
    prev_x, prev_y = None, None
    drawing_mode = False
    drawn_points = []
    piano_drawn = False
    min_x, max_x, min_y, max_y = 0, 0, 0, 0

    print("Press 'd' to toggle drawing mode, 'c' to clear the canvas, and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if canvas is None:
            canvas = np.zeros_like(frame)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                h, w, _ = frame.shape
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                if drawing_mode:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                    prev_x, prev_y = x, y
                    drawn_points.append((x, y))
                elif piano_drawn and abs(x - thumb_x) < 20 and abs(y - thumb_y) < 20:
                    detected_note = detect_note_from_position(x, min_x, max_x, num_keys=7)

                    # Only play note if it hasn't been played yet
                    if not note_playing:
                        # Play the note in a separate thread to avoid blocking
                        play_thread = Thread(target=play_note_in_thread, args=(detected_note,))
                        play_thread.start()

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if not drawing_mode and drawn_points and not piano_drawn:
            min_x = min([p[0] for p in drawn_points])
            max_x = max([p[0] for p in drawn_points])
            min_y = min([p[1] for p in drawn_points])
            max_y = max([p[1] for p in drawn_points])
            canvas = np.zeros_like(frame)
            draw_piano_on_canvas(canvas, min_x, min_y, max_x, max_y)
            piano_drawn = True
            drawn_points.clear()

        overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow(f"Virtual Piano - Camera {camera_index}", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            drawing_mode = False
            piano_drawn = False
            drawn_points.clear()
        elif key == ord('d'):
            drawing_mode = not drawing_mode

    cap.release()
    cv2.destroyAllWindows()

# Tkinter GUI for camera selection and control
def start_camera():
    camera_index = int(dropdown_var.get().split(' ')[1])
    open_camera_with_piano(camera_index)

root = tk.Tk()
root.title("Virtual Piano")

# Dropdown menu to select camera
available_cameras = get_camera_indexes()
dropdown_var = tk.StringVar(root)
dropdown_var.set(f"Camera 0")
camera_options = [f"Camera {i}" for i in available_cameras]
camera_dropdown = ttk.Combobox(root, textvariable=dropdown_var, values=camera_options)
camera_dropdown.pack(pady=20)

# Start button to start the camera
start_button = tk.Button(root, text="Start", command=start_camera)
start_button.pack(pady=20)

root.mainloop()

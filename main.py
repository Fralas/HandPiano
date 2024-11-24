import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk

def get_camera_indexes(max_cameras=5):
    """Return a list of available camera indexes."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def open_camera_with_drawing(camera_index):
    """Open the selected camera and allow drawing on a canvas."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Initialize Mediapipe Hand Solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    # Variables to track drawing state and previous finger position
    canvas = None
    prev_x, prev_y = None, None
    drawing_mode = False  # Start with drawing mode off

    print("Press 'd' to toggle drawing mode, 'c' to clear the canvas, and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip and convert frame to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dynamically initialize the canvas to match the frame size
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Draw mode: Only draw lines if drawing_mode is True
                if drawing_mode and prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                prev_x, prev_y = x, y

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Overlay the canvas on the frame
        overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the combined image
        cv2.imshow(f"Drawing Canvas - Camera {camera_index}", overlay)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Clear the canvas
            canvas = np.zeros_like(frame)
        elif key == ord('d'):  # Toggle drawing mode
            drawing_mode = not drawing_mode
            print("Drawing mode:", "On" if drawing_mode else "Off")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def start_camera():
    """Start the camera feed based on the selected camera index."""
    selected_camera = dropdown_var.get()
    if selected_camera:
        camera_index = int(selected_camera.split()[-1])
        open_camera_with_drawing(camera_index)

# GUI Setup
root = tk.Tk()
root.title("Select Camera")

# Get available cameras
camera_indexes = get_camera_indexes()

if not camera_indexes:
    tk.Label(root, text="No cameras found!").pack(pady=20)
else:
    tk.Label(root, text="Select a camera:").pack(pady=10)

    # Dropdown menu for camera selection
    dropdown_var = tk.StringVar(value=f"Camera {camera_indexes[0]}")
    dropdown = ttk.Combobox(root, textvariable=dropdown_var, state="readonly")
    dropdown['values'] = [f"Camera {index}" for index in camera_indexes]
    dropdown.pack(pady=10)

    # Button to start the camera
    start_button = tk.Button(root, text="Start Camera", command=start_camera)
    start_button.pack(pady=10)

root.mainloop()

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

def draw_piano_on_canvas(canvas, min_x, min_y, max_x, max_y):
    """Draw a piano on the given canvas inside the bounding box."""
    h, w, _ = canvas.shape
    piano_width = max_x - min_x
    piano_height = max_y - min_y

    # Number of white keys (adjust based on the width of the bounding box)
    num_keys = piano_width // 40  # 40 pixels per key is a basic estimation
    key_width = piano_width // num_keys
    key_height = piano_height

    # Draw the white keys (rectangles)
    for i in range(num_keys):
        x_start = min_x + i * key_width
        cv2.rectangle(canvas, (x_start, min_y), (x_start + key_width, min_y + key_height), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x_start, min_y), (x_start + key_width, min_y + key_height), (0, 0, 0), 2)  # Outline of the key

        # Add key labels (C, D, E, F, G, A, B...)
        note = "CDEFGAB"[i % 7]  # Cycle through notes
        cv2.putText(canvas, note, (x_start + 10, min_y + key_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw the black keys (smaller rectangles)
    for i in range(1, num_keys - 1):
        if i % 7 == 2 or i % 7 == 6:  # Skip black key positions (between B-C and E-F)
            continue
        # Calculate position of black key (centered between two white keys)
        x_start = min_x + i * key_width + key_width // 2 - key_width // 4  # Adjust for center
        black_key_width = key_width // 2
        black_key_height = int(key_height // 1.5)

        # Draw the black key as a rectangle
        cv2.rectangle(canvas, (x_start, min_y), (x_start + black_key_width, min_y + black_key_height), (0, 0, 0), -1)

def open_camera_with_piano(camera_index):
    """Open the selected camera and enable piano drawing and interaction."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Initialize Mediapipe Hand Solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    canvas = None
    prev_x, prev_y = None, None
    drawing_mode = False  # Initially, don't start drawing
    drawn_points = []  # To track the points where the user draws
    piano_drawn = False  # Track whether the piano is drawn

    print("Press 'd' to toggle drawing mode, 'c' to clear the canvas, and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip and convert frame to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Initialize the canvas if it's not created yet
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                # Drawing mode: draw the shape with the finger
                if drawing_mode:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                    prev_x, prev_y = x, y

                    # Track drawn points for resizing
                    drawn_points.append((x, y))

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Once drawing mode is off, finalize and resize the drawing
        if not drawing_mode and drawn_points and not piano_drawn:
            # Calculate bounding box (min/max of x and y coordinates)
            min_x = min([p[0] for p in drawn_points])
            max_x = max([p[0] for p in drawn_points])
            min_y = min([p[1] for p in drawn_points])
            max_y = max([p[1] for p in drawn_points])

            # Resize the drawn area into a rectangle
            canvas = np.zeros_like(frame)
            cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), (255, 255, 255), -1)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)

            # Draw the piano inside the bounding box
            draw_piano_on_canvas(canvas, min_x, min_y, max_x, max_y)

            # Finalize drawing the piano shape
            piano_drawn = True
            drawn_points.clear()  # Clear drawn points after finalizing

        # Overlay the canvas on the frame
        overlay = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the frame
        cv2.imshow(f"Virtual Piano - Camera {camera_index}", overlay)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('c'):  # Clear the canvas
            canvas = np.zeros_like(frame)
            drawing_mode = False
            piano_drawn = False
            drawn_points.clear()
            print("Canvas cleared.")
        elif key == ord('d'):  # Toggle drawing mode
            if not drawing_mode:
                drawing_mode = True
                print("Drawing mode: On")
            else:
                drawing_mode = False
                print("Drawing mode: Off\nPiano finalized.")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def start_camera():
    """Start the camera feed based on the selected camera index."""
    selected_camera = dropdown_var.get()
    if selected_camera:
        camera_index = int(selected_camera.split()[-1])
        open_camera_with_piano(camera_index)

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

import cv2
import mediapipe as mp
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

def open_camera_with_hand_detection(camera_index):
    """Open the selected camera and detect hand movements."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    print(f"Displaying camera {camera_index}. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Draw landmarks and connections on the frame if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # Display the frame
        cv2.imshow(f'Camera {camera_index}', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def start_camera():
    """Start the camera feed based on the selected camera index."""
    selected_camera = dropdown_var.get()
    if selected_camera:
        camera_index = int(selected_camera.split()[-1])
        open_camera_with_hand_detection(camera_index)

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

from inference import get_model
import supervision as sv
import cv2
import pyttsx3
import time

# Load the trained YOLO model
model = get_model(model_id="sign-language-detection-obda7/1", api_key="izN7jPjKDsrhBIPIPRw5")

# Initialize annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open webcam (1 = default camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 120)
tts_engine.setProperty('volume', 1.0)

# For preventing repeat speech
last_spoken = {}
speak_interval = 6  # seconds

# FPS calculation
prev_time = time.time()
fps = 0

# Flags for control
capturing = False

print("Press 's' to start capturing, 'e' to stop, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Calculate and update FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    key = cv2.waitKey(1) & 0xFF

    # Start/Stop/Quit logic
    if key == ord('s'):
        print("Started capturing...")
        capturing = True
    elif key == ord('e'):
        print("Stopped capturing.")
        capturing = False
    elif key == ord('q'):
        print("Exiting...")
        break

    display_frame = frame.copy()

    if capturing:
        # Inference
        results = model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        # Labels
        labels = [
            f"{c.class_name} ({c.confidence * 100:.1f}%)"
            for c in results.predictions
        ]

        # Text-to-speech
        for c in results.predictions:
            label = c.class_name
            if label not in last_spoken or (current_time - last_spoken[label]) > speak_interval:
                print(f"Detected: {label} (Confidence: {c.confidence*100:.1f})")
                tts_engine.say(f"{label}")
                tts_engine.runAndWait()
                last_spoken[label] = current_time


        # Annotate
        display_frame = bounding_box_annotator.annotate(scene=display_frame, detections=detections)
        display_frame = label_annotator.annotate(scene=display_frame, detections=detections, labels=labels)

    # Draw FPS at bottom-left
    cv2.putText(
        display_frame,
        f"FPS: {fps:.0f}",
        (10, display_frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Show the frame
    try:
        cv2.imshow("Real-Time Sign Language Translation", display_frame)
    except cv2.error:
        print("[WARNING] Unable to display window. Running in headless mode?")

# Cleanup
cap.release()
cv2.destroyAllWindows()

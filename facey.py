
import cv2
import os

# Step 1: Load Haar Cascade
cascade_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Face Cascade not loaded!")
    exit()

# Step 2: Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not opened!")
    exit()

print("Webcam Started Successfully!")
print("Press C - Capture Photo")
print("Press Q - Quit Webcam")

# Step 3: Create capture folder
os.makedirs("captures", exist_ok=True)

count = 1

# Step 4: Start Webcam
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read webcam!")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Step 5: Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 6: Detect Faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Step 7: Draw Rectangle
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Step 8: Display Face Count
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Step 9: Display Webcam
    cv2.imshow("Face Detection System", frame)

    key = cv2.waitKey(1) & 0xFF

    # Step 10: Capture Photo
    if key == ord('c'):

        filename = f"captures/face_{count}.jpg"

        cv2.imwrite(filename, frame)

        print("Photo Saved:", filename)

        count += 1

    # Step 11: Quit
    elif key == ord('q'):
        print("Webcam Closed!")
        break

# Step 12: Release Resources
cap.release()
cv2.destroyAllWindows()
 
"""
16. Face Detection and Recognition
- Haar Cascade face detection
- Eye, smile detection
- LBP (Local Binary Patterns)
- Face recognition basics
"""

import cv2
import numpy as np


def create_face_image():
    """Simulated face image"""
    img = np.zeros((400, 500, 3), dtype=np.uint8)
    img[:] = [220, 220, 220]

    # Face 1 (left)
    cv2.ellipse(img, (150, 180), (60, 75), 0, 0, 360, (180, 160, 140), -1)
    cv2.circle(img, (130, 160), 10, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (170, 160), 10, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(img, (150, 210), (20, 10), 0, 0, 180, (100, 80, 80), 2)  # Mouth

    # Face 2 (right)
    cv2.ellipse(img, (350, 200), (55, 70), 0, 0, 360, (175, 155, 135), -1)
    cv2.circle(img, (332, 180), 9, (45, 45, 45), -1)  # Left eye
    cv2.circle(img, (368, 180), 9, (45, 45, 45), -1)  # Right eye
    cv2.ellipse(img, (350, 225), (18, 8), 0, 0, 180, (90, 70, 70), 2)  # Mouth

    return img


def haar_cascade_face_detection():
    """Haar Cascade face detection demo"""
    print("=" * 50)
    print("Haar Cascade Face Detection")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade classifier
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Cannot load face detector.")
        return

    # Face detection
    # scaleFactor: Ratio by which image size is reduced at each scale
    # minNeighbors: Number of neighbors required for each candidate rectangle
    # minSize: Minimum object size
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    result = img.copy()
    print(f"Number of faces detected: {len(faces)}")

    for i, (x, y, w, h) in enumerate(faces):
        # Mark face region
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, f'Face {i+1}', (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}")

    print("\nHaar Cascade parameters:")
    print("  scaleFactor: 1.1~1.3 (smaller = more precise, slower)")
    print("  minNeighbors: 3~6 (larger = stricter)")
    print("  minSize: Minimum detection size")

    cv2.imwrite('face_haar_input.jpg', img)
    cv2.imwrite('face_haar_result.jpg', result)


def cascade_eye_detection():
    """Eye detection demo"""
    print("\n" + "=" * 50)
    print("Eye Detection (Haar Cascade)")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load classifiers
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    if face_cascade.empty() or eye_cascade.empty():
        print("Cannot load classifiers.")
        return

    # Face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    result = img.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Detect eyes within face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = result[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        print(f"Eyes detected in face ({x}, {y}): {len(eyes)}")

    print("\nEye detection tips:")
    print("  - Detect only within face region (ROI)")
    print("  - Set minNeighbors high")
    print("  - Searching only upper half improves accuracy")

    cv2.imwrite('face_eye_result.jpg', result)


def available_cascades():
    """Available Haar Cascade list"""
    print("\n" + "=" * 50)
    print("Available Haar Cascade Classifiers")
    print("=" * 50)

    cascades = [
        ('haarcascade_frontalface_default.xml', 'Frontal face'),
        ('haarcascade_frontalface_alt.xml', 'Frontal face (alternative)'),
        ('haarcascade_frontalface_alt2.xml', 'Frontal face (alternative 2)'),
        ('haarcascade_profileface.xml', 'Profile face'),
        ('haarcascade_eye.xml', 'Eye'),
        ('haarcascade_eye_tree_eyeglasses.xml', 'Eye (with glasses)'),
        ('haarcascade_smile.xml', 'Smile'),
        ('haarcascade_upperbody.xml', 'Upper body'),
        ('haarcascade_lowerbody.xml', 'Lower body'),
        ('haarcascade_fullbody.xml', 'Full body'),
    ]

    print(f"\nPath: {cv2.data.haarcascades}\n")

    for filename, description in cascades:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
        status = "OK" if not cascade.empty() else "N/A"
        print(f"  [{status}] {filename}")
        print(f"       - {description}")


def lbp_face_detection():
    """LBP-based face detection demo"""
    print("\n" + "=" * 50)
    print("LBP Face Detection")
    print("=" * 50)

    img = create_face_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load LBP Cascade (if available)
    lbp_cascade_path = cv2.data.haarcascades + '../lbpcascades/lbpcascade_frontalface_improved.xml'

    try:
        lbp_cascade = cv2.CascadeClassifier(lbp_cascade_path)

        if lbp_cascade.empty():
            raise FileNotFoundError

        faces = lbp_cascade.detectMultiScale(gray, 1.1, 5)

        result = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

        print(f"Faces detected with LBP: {len(faces)}")
        cv2.imwrite('face_lbp_result.jpg', result)

    except (FileNotFoundError, cv2.error):
        print("LBP Cascade not found.")
        print("Haar Cascade is recommended in most cases.")

    print("\nHaar vs LBP comparison:")
    print("  Haar: More accurate, slower, sensitive to lighting changes")
    print("  LBP: Faster, robust to lighting changes, lower accuracy")


def face_recognition_concept():
    """Face recognition concept explanation"""
    print("\n" + "=" * 50)
    print("Face Recognition Concepts")
    print("=" * 50)

    # Create test images
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(img1, (50, 50), (30, 40), 0, 0, 360, 150, -1)
    cv2.circle(img1, (40, 45), 5, 50, -1)
    cv2.circle(img1, (60, 45), 5, 50, -1)

    img2 = img1.copy()  # Same person
    img3 = np.zeros((100, 100), dtype=np.uint8)  # Different person
    cv2.ellipse(img3, (50, 50), (35, 35), 0, 0, 360, 160, -1)
    cv2.circle(img3, (35, 45), 6, 60, -1)
    cv2.circle(img3, (65, 45), 6, 60, -1)

    print("Face recognition pipeline:")
    print("  1. Face Detection")
    print("  2. Face Alignment")
    print("  3. Feature Extraction")
    print("  4. Feature Matching")

    print("\nOpenCV face recognizers (opencv-contrib required):")
    print("  - EigenFaces: PCA-based")
    print("  - FisherFaces: LDA-based")
    print("  - LBPH: Local Binary Pattern Histogram")

    # LBPH face recognizer example (opencv-contrib required)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Training data
        faces = [img1, img2, img3]
        labels = np.array([0, 0, 1])  # 0: first person, 1: second person

        recognizer.train(faces, labels)

        # Prediction
        label, confidence = recognizer.predict(img1)
        print(f"\nTest: label={label}, confidence={confidence:.2f}")
        print("  Lower confidence = more similar")

    except AttributeError:
        print("\nNote: To use the LBPH recognizer:")
        print("  pip install opencv-contrib-python")

    cv2.imwrite('face_sample1.jpg', img1)
    cv2.imwrite('face_sample2.jpg', img3)


def face_detection_comparison():
    """Face detection method comparison"""
    print("\n" + "=" * 50)
    print("Face Detection/Recognition Method Comparison")
    print("=" * 50)

    print("""
    | Method | Advantages | Disadvantages | Use Case |
    |--------|-----------|---------------|----------|
    | Haar Cascade | Fast, simple | Weak on profile/tilted | Real-time detection |
    | LBP | Very fast | Low accuracy | Embedded |
    | HOG + SVM | Accurate | Slow | Detection |
    | DNN (SSD) | Very accurate | GPU recommended | High-precision detection |
    | DNN (Face) | Feature extraction | Model required | Recognition |
    """)

    print("Recent trends:")
    print("  - MTCNN: Multi-stage CNN (detection + alignment)")
    print("  - RetinaFace: High-precision detection")
    print("  - ArcFace, FaceNet: Embedding-based recognition")
    print("  - InsightFace: Comprehensive framework")


def real_time_detection_template():
    """Real-time detection template"""
    print("\n" + "=" * 50)
    print("Real-time Face Detection Template")
    print("=" * 50)

    code = '''
# Real-time face detection code
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

    print(code)

    print("Performance optimization tips:")
    print("  1. Frame skipping (not every frame needs detection)")
    print("  2. Downscale image before detection")
    print("  3. Search only around previous detection area")
    print("  4. Use multithreading")


def main():
    """Main function"""
    # Haar Cascade face detection
    haar_cascade_face_detection()

    # Eye detection
    cascade_eye_detection()

    # Available cascade list
    available_cascades()

    # LBP detection
    lbp_face_detection()

    # Face recognition concept
    face_recognition_concept()

    # Method comparison
    face_detection_comparison()

    # Real-time template
    real_time_detection_template()

    print("\nFace detection and recognition demo complete!")


if __name__ == '__main__':
    main()

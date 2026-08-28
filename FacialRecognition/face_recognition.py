''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2

from FacialRecognition import CASCADE_PATH


def load_recognizer(trainer_path="trainer/trainer.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    return recognizer


def build_face_cascade():
    return cv2.CascadeClassifier(CASCADE_PATH)


def recognize_face(recognizer, gray_face, names, confidence_threshold=100):
    """Predict a single grayscale face crop.

    Returns (label, confidence_pct) where label is a name from names or
    "unknown", mirroring the original script's confidence formatting.
    """
    id_, confidence = recognizer.predict(gray_face)
    if confidence < confidence_threshold:
        label = names[id_]
    else:
        label = "unknown"
    return label, round(100 - confidence)


if __name__ == "__main__":
    recognizer = load_recognizer()
    faceCascade = build_face_cascade()

    font = cv2.FONT_HERSHEY_SIMPLEX

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label, confidence_pct = recognize_face(recognizer, gray[y:y + h, x:x + w], names)

            cv2.putText(img, str(label), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, f"  {confidence_pct}%", (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

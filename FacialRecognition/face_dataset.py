''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import os

import cv2

from FacialRecognition import CASCADE_PATH


def build_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


def capture_faces(cam, face_detector, face_id, dataset_dir="dataset", max_count=30,
                   show_window=True):
    """Capture up to max_count face crops from cam and write them to dataset_dir.

    Returns the number of face images written. Kept side-effect-light (no
    ``cv2.imshow``/``cv2.waitKey``) when show_window is False so it can run
    headlessly in tests against a stub camera.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(
                os.path.join(dataset_dir, f"User.{face_id}.{count}.jpg"),
                gray[y:y + h, x:x + w],
            )
            if show_window:
                cv2.imshow('image', img)

        if show_window:
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
        if count >= max_count:  # Take max_count face samples and stop
            break

    return count


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = build_face_detector()

    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    count = capture_faces(cam, face_detector, face_id)

    # Do a bit of cleanup
    print(f"\n [INFO] {count} faces captured. Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

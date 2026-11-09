''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import os

import cv2
import numpy as np
from PIL import Image

from FacialRecognition import CASCADE_PATH


def build_face_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def get_images_and_labels(path, detector):
    """Read dataset images from path, detect the face crop in each, and
    return (face_samples, ids) suitable for LBPHFaceRecognizer.train().

    Non-image files in the dataset directory (e.g. .DS_Store) are skipped
    rather than raising, since the "User.<id>.<count>.<ext>" naming scheme
    is the only place an id is encoded.
    """
    image_paths = [
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(_IMAGE_EXTENSIONS)
    ]
    face_samples = []
    ids = []

    for image_path in image_paths:
        pil_img = Image.open(image_path).convert('L')  # convert it to grayscale
        img_numpy = np.array(pil_img, 'uint8')

        id_ = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id_)

    return face_samples, ids


def train_recognizer(dataset_dir="dataset", trainer_path="trainer/trainer.yml", detector=None):
    """Train an LBPH recognizer on dataset_dir and write it to trainer_path.

    ``detector`` defaults to the package's Haar cascade but can be swapped
    (e.g. in tests) for a stub that skips real face detection, since this
    function's job is the train/write round trip, not detection accuracy.

    Returns (recognizer, num_unique_ids).
    """
    detector = detector or build_face_detector()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = get_images_and_labels(dataset_dir, detector)
    recognizer.train(faces, np.array(ids))

    os.makedirs(os.path.dirname(trainer_path) or ".", exist_ok=True)
    recognizer.write(trainer_path)  # recognizer.save() worked on Mac, but not on Pi

    return recognizer, len(np.unique(ids))


if __name__ == "__main__":
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    _, num_ids = train_recognizer()
    print(f"\n [INFO] {num_ids} faces trained. Exiting Program")

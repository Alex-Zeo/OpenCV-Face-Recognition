"""Headless smoke tests for the S1 refactor: the three FacialRecognition
pipeline stages must import with no webcam, no input() prompt and no
window, and resolve the cascade asset from the package rather than the cwd.
"""
import importlib
import os

import cv2
import pytest

import FacialRecognition
from FacialRecognition import CASCADE_PATH


def test_cascade_path_resolves_from_package_not_cwd():
    assert os.path.dirname(CASCADE_PATH) == FacialRecognition.PACKAGE_DIR
    assert os.path.isfile(CASCADE_PATH)


def test_cascade_classifier_loads_from_package_path():
    classifier = cv2.CascadeClassifier(CASCADE_PATH)
    assert not classifier.empty()


@pytest.mark.parametrize("module_name", [
    "FacialRecognition.face_dataset",
    "FacialRecognition.face_training",
    "FacialRecognition.face_recognition",
])
def test_stage_module_imports_headlessly(module_name):
    # Importing must not open a camera, block on input(), or open a window;
    # that behavior is guarded under `if __name__ == "__main__":` in each
    # stage module, so a plain import returning is the assertion itself.
    module = importlib.import_module(module_name)
    assert module is not None


def test_face_dataset_builds_detector_from_package_cascade():
    from FacialRecognition.face_dataset import build_face_detector
    detector = build_face_detector()
    assert not detector.empty()


def test_face_training_builds_detector_from_package_cascade():
    from FacialRecognition.face_training import build_face_detector
    detector = build_face_detector()
    assert not detector.empty()


def test_get_images_and_labels_handles_empty_dataset_dir(tmp_path):
    from FacialRecognition.face_training import build_face_detector, get_images_and_labels
    faces, ids = get_images_and_labels(str(tmp_path), build_face_detector())
    assert faces == []
    assert ids == []

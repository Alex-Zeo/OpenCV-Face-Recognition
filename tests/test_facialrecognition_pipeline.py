"""S2 round-trip coverage: dataset writer -> LBPH train -> recognizer,
run headlessly end to end with synthetic images and a stub detector.

Real face detection is already covered by
test_facialrecognition_imports.py (cascade-asset load); these tests
inject a stub detector so identity discrimination -- the thing LBPH
train/predict actually need to prove out -- isn't gated on the fixture
images being photographically detectable as faces. Bundling a real,
license-clean fixture gallery for detection+accuracy is S3's job.
"""
import os

import cv2
import numpy as np

from FacialRecognition.face_dataset import capture_faces
from FacialRecognition.face_recognition import load_recognizer, recognize_face
from FacialRecognition.face_training import get_images_and_labels, train_recognizer


class _StubDetector:
    """detectMultiScale stand-in: always reports one fixed box."""

    def __init__(self, box):
        self._box = box

    def detectMultiScale(self, image, *args, **kwargs):
        return [self._box]


class _StubCamera:
    """cam.read() stand-in that replays a fixed list of frames, then EOF."""

    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)


def _make_identity_image(identity_id, sample_idx, size=80):
    """A deterministic, per-identity textured grayscale crop.

    Different identities get different shapes (filled rectangle+circle
    vs. stripes+circle) so LBPH's local-texture histograms separate
    them; a small per-sample noise jitter (seeded on sample_idx) keeps
    samples of the same identity distinct without changing the texture.
    """
    rng = np.random.default_rng(1000 * identity_id + sample_idx)
    img = np.zeros((size, size), dtype=np.uint8)
    if identity_id == 1:
        cv2.rectangle(img, (10, 10), (70, 70), 200, -1)
        cv2.circle(img, (40, 40), 15, 60, -1)
    else:
        for y in range(0, size, 8):
            cv2.line(img, (0, y), (size, y), 180, 2)
        cv2.circle(img, (40, 40), 25, 40, 3)
    noise = rng.integers(-10, 10, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_capture_faces_writes_dataset_images_from_stub_camera(tmp_path):
    dataset_dir = tmp_path / "dataset"
    frames = [
        cv2.cvtColor(_make_identity_image(1, i), cv2.COLOR_GRAY2BGR)
        for i in range(5)
    ]
    cam = _StubCamera(frames)
    detector = _StubDetector((0, 0, 80, 80))

    count = capture_faces(
        cam, detector, face_id=1, dataset_dir=str(dataset_dir),
        max_count=5, show_window=False,
    )

    assert count == 5
    written = sorted(os.listdir(dataset_dir))
    assert written == [f"User.1.{n}.jpg" for n in range(1, 6)]
    # each file is the flipped, gray-cropped face region, not a raw copy
    sample = cv2.imread(str(dataset_dir / "User.1.1.jpg"), cv2.IMREAD_GRAYSCALE)
    assert sample.shape == (80, 80)


def test_capture_faces_stops_when_camera_runs_out_of_frames(tmp_path):
    dataset_dir = tmp_path / "dataset"
    cam = _StubCamera([cv2.cvtColor(_make_identity_image(1, 0), cv2.COLOR_GRAY2BGR)])
    detector = _StubDetector((0, 0, 80, 80))

    count = capture_faces(
        cam, detector, face_id=1, dataset_dir=str(dataset_dir),
        max_count=30, show_window=False,
    )

    assert count == 1  # camera EOF, not max_count, ended the loop


def _write_dataset(dataset_dir, identity_id, num_samples):
    dataset_dir.mkdir(exist_ok=True)
    for i in range(num_samples):
        img = _make_identity_image(identity_id, i)
        cv2.imwrite(str(dataset_dir / f"User.{identity_id}.{i}.jpg"), img)


def test_get_images_and_labels_pairs_each_face_with_its_id(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, identity_id=1, num_samples=3)
    _write_dataset(dataset_dir, identity_id=2, num_samples=2)

    faces, ids = get_images_and_labels(str(dataset_dir), _StubDetector((0, 0, 80, 80)))

    assert len(faces) == len(ids) == 5
    assert sorted(ids) == [1, 1, 1, 2, 2]


def test_train_recognizer_round_trip_writes_trainer_file(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, identity_id=1, num_samples=5)
    _write_dataset(dataset_dir, identity_id=2, num_samples=5)
    trainer_path = tmp_path / "trainer" / "trainer.yml"

    recognizer, num_ids = train_recognizer(
        dataset_dir=str(dataset_dir),
        trainer_path=str(trainer_path),
        detector=_StubDetector((0, 0, 80, 80)),
    )

    assert num_ids == 2
    assert trainer_path.is_file()
    assert isinstance(recognizer, cv2.face.LBPHFaceRecognizer)


def test_recognizer_round_trip_labels_held_out_crops_by_identity(tmp_path):
    # Train on samples 0..4 per identity; predict on sample 99, which was
    # never part of the training set, so this can't pass by memorizing.
    dataset_dir = tmp_path / "dataset"
    _write_dataset(dataset_dir, identity_id=1, num_samples=5)
    _write_dataset(dataset_dir, identity_id=2, num_samples=5)
    trainer_path = tmp_path / "trainer" / "trainer.yml"
    train_recognizer(
        dataset_dir=str(dataset_dir),
        trainer_path=str(trainer_path),
        detector=_StubDetector((0, 0, 80, 80)),
    )

    recognizer = load_recognizer(str(trainer_path))
    names = ["None", "Alice", "Bob"]

    label_1, confidence_1 = recognize_face(recognizer, _make_identity_image(1, 99), names)
    label_2, confidence_2 = recognize_face(recognizer, _make_identity_image(2, 99), names)

    assert label_1 == "Alice"
    assert label_2 == "Bob"
    assert confidence_1 > 0
    assert confidence_2 > 0


def test_recognize_face_reports_unknown_below_confidence_threshold():
    class _AlwaysLowConfidence:
        def predict(self, gray_face):
            return 1, 150  # confidence >= default threshold of 100

    label, confidence_pct = recognize_face(
        _AlwaysLowConfidence(), np.zeros((10, 10), dtype=np.uint8), ["None", "Alice"],
    )

    assert label == "unknown"
    assert confidence_pct == -50

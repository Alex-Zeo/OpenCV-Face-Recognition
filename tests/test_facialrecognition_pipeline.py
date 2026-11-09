"""S2 coverage: the dataset writer, LBPH train, and recognizer round-trip,
all exercised headlessly with synthetic images and a stub detector so no
webcam, real face, or display is needed.
"""
import numpy as np
import pytest

from FacialRecognition.face_dataset import capture_faces
from FacialRecognition.face_recognition import load_recognizer, recognize_face
from FacialRecognition.face_training import train_recognizer


class _StubCam:
    """Fakes cv2.VideoCapture: yields a fixed number of frames, then ret=False."""

    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)


class _StubDetector:
    """Fakes cv2.CascadeClassifier.detectMultiScale: always reports the
    whole image as a single face box, so tests don't depend on a real
    face being present in the synthetic frame."""

    def detectMultiScale(self, image, *args, **kwargs):
        h, w = image.shape[:2]
        return [(0, 0, w, h)]


def _solid_bgr_frame(value, size=40):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _solid_gray_image(value, size=40):
    return np.full((size, size), value, dtype=np.uint8)


def test_capture_faces_writes_cropped_dataset_images(tmp_path):
    frames = [_solid_bgr_frame(50), _solid_bgr_frame(60), _solid_bgr_frame(70)]
    cam = _StubCam(frames)
    dataset_dir = tmp_path / "dataset"

    count = capture_faces(
        cam, _StubDetector(), face_id=7, dataset_dir=str(dataset_dir),
        max_count=3, show_window=False,
    )

    assert count == 3
    written = sorted(p.name for p in dataset_dir.iterdir())
    assert written == ["User.7.1.jpg", "User.7.2.jpg", "User.7.3.jpg"]


def test_capture_faces_stops_when_camera_runs_out_of_frames(tmp_path):
    cam = _StubCam([_solid_bgr_frame(80)])
    dataset_dir = tmp_path / "dataset"

    count = capture_faces(
        cam, _StubDetector(), face_id=1, dataset_dir=str(dataset_dir),
        max_count=30, show_window=False,
    )

    assert count == 1
    assert len(list(dataset_dir.iterdir())) == 1


@pytest.fixture
def two_id_dataset(tmp_path, monkeypatch):
    """A synthetic dataset with two distinct ids, two images each, plus a
    stubbed detector (patched into face_training) so the fixture is not
    dependent on a real face being detectable in synthetic pixels."""
    from PIL import Image

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for count in (1, 2):
        Image.fromarray(_solid_gray_image(30)).save(dataset_dir / f"User.1.{count}.jpg")
        Image.fromarray(_solid_gray_image(220)).save(dataset_dir / f"User.2.{count}.jpg")

    from FacialRecognition import face_training
    monkeypatch.setattr(face_training, "build_face_detector", lambda: _StubDetector())
    return dataset_dir


def test_train_recognizer_trains_on_dataset_and_writes_trainer_file(tmp_path, two_id_dataset):
    trainer_path = tmp_path / "trainer" / "trainer.yml"

    _recognizer, num_ids = train_recognizer(
        dataset_dir=str(two_id_dataset), trainer_path=str(trainer_path),
    )

    assert num_ids == 2
    assert trainer_path.is_file()


def test_recognizer_round_trip_predicts_trained_id(tmp_path, two_id_dataset):
    trainer_path = tmp_path / "trainer" / "trainer.yml"
    train_recognizer(dataset_dir=str(two_id_dataset), trainer_path=str(trainer_path))

    loaded = load_recognizer(trainer_path=str(trainer_path))
    names = ["None", "Marcelo", "Paula"]

    label, confidence_pct = recognize_face(loaded, _solid_gray_image(30), names)

    assert label == "Marcelo"
    assert confidence_pct > 0

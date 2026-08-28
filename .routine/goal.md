<!-- .routine/goal.md -->
# facial-rec — OpenCV LBPH face-recognition goal

North-star: turn the Mjrovai OpenCV tutorial (seven camera- and display-bound
scripts: Haar-cascade face/eye/smile detection plus the three-stage LBPH
dataset -> train -> recognize pipeline) into a reproducible, importable,
test-covered package whose recognition quality can be measured in CI without a
webcam attached.

Current state (2026-08-28): working tutorial-grade scripts, but no packaging
(no requirements/pyproject), no importable modules (every script opens
`cv2.VideoCapture(0)` and calls `input()` at import time, and hardcodes the
cascade path so it only runs from inside `FacialRecognition/`), zero tests, and
no CI. The pinned model asset is `haarcascade_frontalface_default.xml`; the
recognizer is `cv2.face.LBPHFaceRecognizer_create()` from opencv-contrib.

The bootstrap sub-objectives below are the genuine next engineering steps: make
it installable and importable, stand up a headless test suite, and pin a real
recognition-accuracy floor measured on a held-out split.

```toml
axis_priority = ["correctness", "user-value", "quality", "perf"]

[thesis]
beneficiary = "developers and makers building on the OpenCV LBPH face-recognition pipeline, who today can only run it live at a webcam"
counterfactual = "without this the pipeline stays camera- and display-bound tutorial scripts, uninstallable, unimportable and untested, so recognition quality can never be verified in CI and no one can safely build on it"

[[sub_objective]]
id = "S1"
text = "Reproducible env + importable pipeline: pin deps in requirements.txt and refactor the three FacialRecognition stages so each imports headlessly (camera/input()/display guarded under __main__) and resolves the cascade path from the package, not the cwd"
metric = "importable_pipeline_modules"
threshold = 3
criterion = "imports-clean"

[[sub_objective]]
id = "S2"
text = "First green headless test suite: add pytest covering cascade-asset load, the dataset writer, LBPH train, and recognizer round-trip, all runnable with no webcam or display"
metric = "tests_passing"
threshold = 6
criterion = "pytest-green"

[[sub_objective]]
id = "S3"
text = "Recognition-accuracy floor: on a held-out split of a small bundled fixture gallery, the trained LBPH model must label held-out crops at or above the floor (train and evaluate on disjoint images so the score cannot be met by predicting on training data)"
metric = "recognition_accuracy_pct"
threshold = 85
criterion = "accuracy-gate"

[[abort]]
metric = "cascade_asset_changed"
op = "=="
value = true
```

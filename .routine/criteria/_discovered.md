<!-- .routine/criteria/_discovered.md -->
# facial-rec — discovered deterministic gates + quality criteria

The repo ships no gate config today (no requirements/pyproject, no tests, no
CI). These are the deterministic gates the routine establishes and runs, in the
Python discovery order (requirements -> install, headless import smoke,
`pytest`, and the accuracy check), one llm-as-a-verifier entry each. A gate is a
hard pass/fail; a gate ERROR is a hard stop, never a silent pass.

## Group A — deterministic gates (these authorize a change)
- **install-succeeds**: Does `python -m pip install -r requirements.txt` complete with exit 0, pinning opencv-contrib-python (for `cv2.face`), numpy and Pillow, so the environment is reproducible from the manifest alone?
- **imports-clean**: Does each of the three FacialRecognition pipeline stages import with no webcam, no `input()` prompt and no window, resolving the cascade asset from the package rather than the current directory (metric `importable_pipeline_modules` == 3)?
- **pytest-green**: Does `pytest -q` exit 0 with the whole suite passing headlessly, with no `cv2.VideoCapture`/`imshow`/`waitKey` reached during collection or run (metric `tests_passing` >= 6)?
- **accuracy-gate**: On a disjoint train/held-out split of the bundled fixture gallery, does the trained LBPH model score `recognition_accuracy_pct` >= 85 on held-out crops, computed by the test itself so it cannot pass by evaluating on training images?

## Group B — quality criteria (advisory; no deterministic gate)
- **mergeable-value**: Does the fire land one concrete, operator-mergeable artifact that advances exactly one unmet sub-objective and passes every Group A gate it touches — not a heartbeat-only or busywork commit?
- **no-regression**: Does the change leave the other axes (correctness, user-value, quality, perf) no worse — the pinned `haarcascade_frontalface_default.xml` asset unchanged, previously green gates still green, and no already-importable module made unimportable?

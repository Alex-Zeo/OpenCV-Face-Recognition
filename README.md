# OpenCV-Face-Recognition
Real-time face recognition project with OpenCV and Python
<br><br>
## Running the pipeline

`FacialRecognition/` is an importable package (`FacialRecognition/__init__.py`), and each
stage script resolves the Haar cascade via `from FacialRecognition import CASCADE_PATH`.
Run each stage as a module **from the repo root** so that import resolves:

```bash
pip install -r requirements.txt
python -m FacialRecognition.face_dataset      # capture a new face dataset
python -m FacialRecognition.face_training     # train the LBPH recognizer
python -m FacialRecognition.face_recognition  # run live recognition
```

Running a stage by file path instead (e.g. `python FacialRecognition/face_recognition.py`)
will fail with `ModuleNotFoundError: No module named 'FacialRecognition'`, since the script's
own directory — not the repo root — would be on `sys.path`.
<br><br>
Links for complete Tutorial:
<br>
https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826
https://www.instructables.com/id/Real-time-Face-Recognition-an-End-to-end-Project/
<br>
<p><img src="https://github.com/Mjrovai/OpenCV-Face-Recognition/blob/master/FaceRecogBlock.png?raw=true"></p>

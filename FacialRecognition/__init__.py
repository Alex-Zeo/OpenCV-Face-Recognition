"""Importable LBPH face-recognition pipeline (dataset -> train -> recognize).

Camera capture, ``input()`` prompts and display windows only run when a
stage module is executed as a script (``if __name__ == "__main__":``), so
every module here can be imported headlessly for testing.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(PACKAGE_DIR, "haarcascade_frontalface_default.xml")

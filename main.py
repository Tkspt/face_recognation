from classes.ml_face_recognizer import MlFaceRecognizer
from script import open_on_web, open_on_console

recognizer = MlFaceRecognizer()

# recognizer.train('images/train', 'images/val')
# recognizer.evaluate('images/test')
# recognizer.save("models/ml.h5")

recognizer.read("models/ml.h5")
open_on_console(recognizer, 'evans.png')



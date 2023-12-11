from classes.ml_face_recognizer import MlFaceRecognizer

recognizer = MlFaceRecognizer()
tester = MlFaceRecognizer()

# recognizer.train('images/train', 'images/val')
# recognizer.evaluate('images/test')
# recognizer.save("models/ml.h5")

tester.read("models/ml.h5")
class_name, conf_score = tester.predict('evans.png')
print("## {}".format(class_name))
print("### score: {}%".format(int(conf_score * 1000) / 10))



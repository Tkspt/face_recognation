from classes.ml_face_recognizer import MlFaceRecognizer

recognizer = MlFaceRecognizer()
tester = MlFaceRecognizer()

recognizer.prepare_data('images/train', 'images/val')
recognizer.create_model()
recognizer.train_model()
recognizer.evaluate_model('images/test')
recognizer.save("models/lm.h5")
recognizer.read("models/lm.h5")
recognizer.predict('images/test')



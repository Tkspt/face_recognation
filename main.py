from classes.ml_face_recognizer import MlFaceRecognizer
from script import open_on_web, open_on_console

def create_new_model(train_data_path, val_data_path, test_data_path, model_name): 
    recognizer = MlFaceRecognizer()
    recognizer.train(train_data_path, val_data_path)
    recognizer.evaluate(test_data_path)
    recognizer.save(f"models/{model_name}.h5")
    print("\nNew model saved successfully\n")


def use_model(model_name, open_type, img_file):
    model_path = f"models/{model_name}.h5"
    recognizer = MlFaceRecognizer()
    recognizer.read(model_path)
    
    if open_type == "web":
        open_on_web(recognizer, img_file)
    elif open_type == "cam":
        print("in building")
    elif open_type == "console":
        open_on_console(model_path, img_file)
    else:
        print("Open mode not includ")


# create_new_model('images/train', 'images/val', 'images/test', 'avengers')
use_model("avengers", "web", "bgr.png")

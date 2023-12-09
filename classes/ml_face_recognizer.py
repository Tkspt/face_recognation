import os
import numpy as np
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

class MlFaceRecognizer:
    def __init__(self):
      self.model = None
      self.train_generator = None
      self.val_generator = None
      
    def prepare_data(self, train_data_path, validation_data_path = None):
        # Récupération des chemins des données ...
        current_dir = os.getcwd()
        train_dir = os.path.join(current_dir, train_data_path)
        val_dir = os.path.join(current_dir, validation_data_path)
        
        # Préparation des générateurs ...
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Récupération des données de train et test et val ...
        batch_size = 120
        target_size = (220, 220)

        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
        )

        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
        )
        
        print(self.train_generator)
        print('\nSome lines\n')
        print(self.val_generator)
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(220, 220, 3)))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        
        self.model = model
    
    def train_model(self):
        # Compilation du modèle ...
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entrainement du modèle ...
        epochs = 20
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator
        )
        
        # Visualisons les loss d'entrainement et de validation ...
        plt.figure(figsize=(14, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
    def evaluate_model(self, test_data_path):
        batch_size = 120
        target_size = (220, 220)
        
        # Récupération du chemin des données de test ...
        current_dir = os.getcwd()
        test_dir = os.path.join(current_dir, test_data_path)
        
        # Préparation du générateur ...
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Récupération des données de test ...
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
        )
        
        # Evaluation du modèle ...
        score = self.model.evaluate(test_generator, verbose=0)
        print(f'Test loss     : {score[0]:4.4f}')
        print(f'Test accuracy : {score[1]:4.4f}')
        
    def train(self, train_data_path, validation_data_path):
        self.prepare_data(train_data_path, validation_data_path)
        self.create_model()
        self.train_model()
      
    def save(self, path):
        self.model.save(path)
    
    def read(self, path):
        # load specifical model 
        self.model = load_model(path, compile=False)
        print("done!!!")
    
    def predict(self, image):
        # load class names
        with open('./models/labels.txt', 'r') as f:
            class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
            f.close()
            
        # convert image to (224, 224)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        # convert image to numpy array
        image_array = np.asarray(image)

        # normalize image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # set model input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # make prediction
        prediction = self.model.predict(data)

        index = np.argmax(prediction)
        
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        return class_name, confidence_score
        
        # classify image
        label, score = classify(image, model, class_names)
        
        return (label, score)
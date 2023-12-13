import os
import numpy as np
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt
from keras.optimizers import Adam

class MlFaceRecognizer:
    def __init__(self):
      self.model = None
      self.train_generator = None
      self.val_generator = None
      self.tuner = None
      self.class_names = []
      
    def _prepare_data(self, train_data_path, validation_data_path = None):
        # Récupération des chemins des données ...
        current_dir = os.getcwd()
        train_dir = os.path.join(current_dir, train_data_path)
        val_dir = os.path.join(current_dir, validation_data_path)
        
        self.class_names = os.listdir(train_dir)
        
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
        
        print("data preparation complete ....")
    
    def _create_model(self):
        def build_model(hp):
            output = len(self.class_names)
            nb_neurons1 = hp.Int('nb_neurons1', min_value=20, max_value=128, step=20)
            nb_neurons2 = hp.Int('nb_neurons2', min_value=20, max_value=128, step=20)
            dense_neurons = hp.Int('dense_neurons', min_value=100, max_value=300, step=20)
            epochs = hp.Int('epochs', min_value=5, max_value=20, step=1, default=10)
            learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
            has_dropout = hp.Boolean("dopout")
            if has_dropout :
                dropout = hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)
            
            Adam(learning_rate=learning_rate)
            
            model = Sequential()
            model.add(Conv2D(nb_neurons1, (3, 3), activation='relu', input_shape=(220, 220, 3)))
            if has_dropout:
                model.add(Dropout(dropout))
            model.add(Conv2D(nb_neurons2, (3, 3), activation='relu'))
            if has_dropout:
                model.add(Dropout(dropout))
            model.add(Flatten())
            model.add(Dense(dense_neurons, activation='relu'))
            model.add(Dense(output, activation='softmax'))
            
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            return model

        epochs = 20
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        # Instancions le tuner
        tuner = kt.RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=3,
            overwrite = True,
            directory='tuner_logs',
            project_name='image_recognation'
        )

        # Exécutons la recherche des paramètres
        tuner.search(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=[early_stopping]
        )
        
        best_hps = tuner.get_best_hyperparameters()[0]

        self.tuner = tuner
        self.model = tuner.hypermodel.build(best_hps)
        print("model creation complete ....")

    # def _create_model(self):
    #     output = len(self.class_names)
    #     model = Sequential()
    #     model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(220, 220, 3)))
    #     model.add(MaxPooling2D((2, 2)))

    #     model.add(Conv2D(32, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D((2, 2)))

    #     model.add(Conv2D(64, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D((2, 2)))

    #     model.add(Flatten())
    #     model.add(Dense(100, activation='relu'))
    #     model.add(Dense(output, activation='softmax'))
        
    #     self.model = model
        
    #     print("model creation complete ....")
           
    def _train_model(self):
        best_hps = self.tuner.get_best_hyperparameters()[0]
        
        # Entrainement du modèle ...
        # epochs = 20
        epochs = best_hps['epochs']
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
        
        print("model training complete ....")
    
    def evaluate(self, test_data_path):
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
        
        print("\model evaluation complete ....")
        
    def train(self, train_data_path, validation_data_path):
        self._prepare_data(train_data_path, validation_data_path)
        self._create_model()
        self._train_model()
      
    def save(self, model_path):
        path = model_path.split('/')[0]        
        model_name = model_path.split('/')[-1].split('.')[0]
        
        with open(f'{path}/labels_{model_name}.txt', 'w') as f:
            for index, label in enumerate(self.class_names):
                f.write(f'{index} {label}\n')
            f.close()
        
        self.model.save(model_path)
    
    def read(self, model_path):
        path = model_path.split('/')[0]        
        model_name = model_path.split('/')[-1].split('.')[0]
        
        # load class names
        with open(f'{path}/labels_{model_name}.txt', 'r') as f:
            self.class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
            f.close()
        
        # load specifical model 
        self.model = load_model(model_path, compile=False)
        print("done!!!")
    
    def predict(self, image_path):
            
        # Open the image   
        image = Image.open(image_path).convert('RGB')
            
        # convert image to (220, 220)
        image = ImageOps.fit(image, (220, 220), Image.Resampling.LANCZOS)

        # convert image to numpy array
        image_array = np.asarray(image)

        # normalize image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # set model input
        data = np.ndarray(shape=(1, 220, 220, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # make prediction
        prediction = self.model.predict(data)

        index = np.argmax(prediction)
        
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]

        return class_name, confidence_score
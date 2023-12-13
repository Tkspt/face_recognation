from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
import tensorflow as tf




def build_lstm_model(hp):
    model = Sequential()
    nb_neurons1 = hp.Int('nb_neurons1', min_value=20, max_value=128, step=20)
    nb_neurons2 = hp.Int('nb_neurons2', min_value=20, max_value=128, step=20)
    dense_neurons = hp.Int('dense_neurons', min_value=100, max_value=300, step=20)
    dropout = hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)
    has_dropout = hp.Boolean("dopout")
    epochs = hp.Int('epochs', min_value=5, max_value=15, step=1, default=10)
    learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
    output = 5

    # LSTM layers with hyperparameter tuning
    model.add(Conv2D(nb_neurons1, return_sequences=True))

    if has_dropout:
        model.add(Dropout(dropout))

    model.add(Conv2D(nb_neurons2))

    if has_dropout:
        model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(dense_neurons, activation='relu'))

    model.add(Dense(output, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        metrics=['accuracy'],
    )

    return model

tuner = RandomSearch(
    build_lstm_model,
    objective='val_accuracy',
    max_trials=3,
    directory='tuner_logs',
    project_name='my_text_classification'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=64)
best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)
# model = build_lstm_model(best_hps)

# Obtention des meilleurs hyperparamètres trouvés par Keras Tuner
# Création d'une instance du modèle avec les paramètres optimaux

# Entraînement du modèle avec les données d'entraînement et de validation
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    epochs=best_hps['epochs'],
    callbacks=[early_stopping] # Utilisez des callbacks tels que EarlyStopping pour arrêter l'entraînement lorsque la performance sur les données de validation cesse de s'améliorer.
)
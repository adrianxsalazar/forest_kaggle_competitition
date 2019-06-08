import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder
from numpy import argmax

def artificial_neural_net(number_attributes, number_output,attributes_training,\
labels_training, attributes_testing, labels_testing, data_to_predic ,accuracy=True,\
prediction=False):
    #Simple artificial neural network
    #Transform the labels into an ann compatible format
    ann_training_labels=OneHotEncoder().fit_transform(labels_training[:, np.newaxis]).toarray()
    ann_testing_labels=OneHotEncoder().fit_transform(labels_testing[:, np.newaxis]).toarray()

    #Add layers. The input layer contains as many as features of the input.
    #The output layer contains as many as neurons as the number of different\
    #classes
    artificial_neural_network_model = keras.Sequential()
    artificial_neural_network_model.add(keras.layers.Dense(124,input_dim=number_attributes,\
    activation='relu'))
    artificial_neural_network_model.add(keras.layers.Dense(64, activation='relu'))
    artificial_neural_network_model.add(keras.layers.Dense(number_output,\
    activation='softmax'))

    #Compile the model
    artificial_neural_network_model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #train the models
    artificial_neural_network_model.fit(attributes_training,ann_training_labels,\
    validation_data=(attributes_testing,ann_testing_labels),epochs=150, verbose=0)

    #Evaluaate the model
    perf=artificial_neural_network_model.\
    evaluate(attributes_testing,ann_testing_labels,verbose=0)

    if accuracy == True:
        return perf[1]

    if prediction ==True:
        prediction=artificial_neural_network_model.predict_classes(data_to_predic)
        return prediction
    else:
        return artificial_neural_network_model

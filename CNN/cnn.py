#convolutional neural network
#part-1:Building CNN
#___________________________________________________
#importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initalizing The CNN
classifier=Sequential()
#Step-1:Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Step-2:Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step-3:Flattening
classifier.add(Flatten())
#step-4:Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
#compling
classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])

#PART-2:fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/home/meet-cric/Desktop/WorkSpace/ML/deep learning/Deep Learning A-Z/CNN/dataset/train_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
testing_data = test_datagen.flow_from_directory(
        '/home/meet-cric/Desktop/WorkSpace/ML/deep learning/Deep Learning A-Z/CNN/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

from PIL import Image
classifier.fit_generator(
        training_set,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=testing_data,
        nb_val_samples=2000)
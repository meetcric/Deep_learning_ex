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

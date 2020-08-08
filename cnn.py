#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

#step1 Initializing CNN
classifier = Sequential()

# step2 adding 1st Convolution layer and Pooling layer
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step3 adding 2nd convolution layer and polling layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


#step4 Flattening the layers
classifier.add(Flatten())

#step5 Full_Connection

classifier.add(Dense(output_dim=32,activation = 'relu'))

classifier.add(Dense(output_dim=64,activation = 'relu'))

classifier.add(Dense(output_dim=128,activation = 'relu'))

classifier.add(Dense(output_dim=256,activation = 'relu'))

classifier.add(Dense(output_dim=256,activation = 'relu'))

classifier.add(Dense(output_dim=6,activation = 'softmax'))

#step5 Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#step6 Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                target_size=(64,64),
                                                batch_size=10,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),
                                            batch_size=10,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch=1212,
                         nb_epoch = 20,
                         validation_data =test_set,
                         nb_val_samples = 300)

#step7 saving model 
classifier.save("model.h5")




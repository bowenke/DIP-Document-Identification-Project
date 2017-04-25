import sys
import keras.preprocessing.image as ppimage

def preprocessing_function_1(image):
    return image

# change preprocessing function
train_datagen = ppimage.ImageDataGenerator(
    preprocessing_function=None)

# change preprocessing function
test_datagen = ppimage.ImageDataGenerator(
    preprocessing_function=None)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    color_mode='grayscale',
    batch_size=32,
    class_mode=None)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    color_mode='grayscale',
    batch_size=32,
    class_mode=None)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800)

import _collections as col
import numpy as np
import time
from keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.models import Sequential, Model
from keras.preprocessing import sequence as seq
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from skimage import io


def reduced_model():
    pass


# Build and return a large, full model with ConvNet and attention-based RNN enc-dec
def full_model(num_of_class=-1):
    print('Building model...')

    conv_net = Sequential()

    # layer 1
    conv_net.add(Conv2D(256,
                        data_format='channels_last',
                        input_shape=(None, None, 1),
                        kernel_size=(3, 3),
                        activation='relu',
                        ))
    conv_net.add(BatchNormalization())

    # layer 2
    conv_net.add(Conv2D(256,
                        kernel_size=(3, 3),
                        activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 3
    conv_net.add(Conv2D(64,
                        kernel_size=(3, 3),
                        activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 4
    conv_net.add(Conv2D(64,
                        kernel_size=(3, 3),
                        activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 5
    conv_net.add(Conv2D(32,
                        kernel_size=(3, 3),
                        activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    enc_dec = Sequential()
    # Encoder
    enc_dec.add(Flatten())
    lstm = LSTM(32, activation='relu', input_shape=(None, None, 1))
    enc_dec.add(Bidirectional(lstm))

    # Decoder
    enc_dec.add(LSTM(256, activation='relu'))
    enc_dec.add(Dense(num_of_class))
    enc_dec.add(Activation('softmax'))

    _model = Model(inputs=conv_net.input, outputs=enc_dec(conv_net.output))
    return _model


# Split data into training, testing and validation sets and load them
def split_dataset_rand(paths, labels, train_prop=0.85, test_prop=0.1):
    print("Preparing training, testing and validation datasets...")
    if len(paths) != len(labels):
        print("Fatal: data missing and not aligned")
    indices = np.arange(start=0, stop=len(paths), dtype=np.int64)
    np.random.shuffle(indices)
    leng = len(paths)
    train_size = int(0.85*leng)
    test_size = int(0.1*leng)
    training_indices = indices[:train_size]
    testing_indices = indices[train_size:train_size+test_size]
    validation_indices = indices[train_size+test_size:]

    Xtrain = preprocess_images([paths[i] for i in training_indices])
    Ytrain = [labels[i] for i in training_indices]
    Xtest = preprocess_images([paths[i] for i in testing_indices])
    Ytest = [labels[i] for i in testing_indices]
    Xval = None
    Yval = None
    if leng > train_size+test_size:
        Xval = preprocess_images([paths[i] for i in validation_indices])
        Yval = [labels[i] for i in validation_indices]
    print("Training Set: {} | Testing Set: {} | Validation Set: {}".format(train_size, test_size, leng-(train_size+test_size)))
    return Xtrain, Ytrain, Xtest, Ytest, Xval, Yval


# Read dataset info script and generate labels and paths to images
def preprocess_labels():
    print("Encoding labels and parsing paths to images...")
    char_set = set()
    dic = col.defaultdict(int)
    sentences = []
    paths = []
    with open("lines.txt") as infile:
        for line in infile:
            line_list = (line.split())
            output = line_list[-1].replace('|', ' ')
            li = list(output)
            li = ['<start>'] + li + ['<end>']
            paths.append(line_list[0])
            sentences.append(li)
            for char in output:
                char_set.add(char)
    count = 1
    for char in char_set:
        if dic[char] == 0:
            dic[char] = count
            count += 1
    dic['<end>'] = 0
    dic['<start>'] = count
    ys = []
    maxlen = 0
    for sentence in sentences:
        leng = len(sentence)
        if leng > maxlen: maxlen = leng
        y = np.zeros(shape=(leng, count+1))
        for i in range(leng):
            y[i, dic[sentence[i]]] = 1
        ys.append(y)
    labels = seq.pad_sequences(ys, maxlen, 'int64', 'post')
    leng = len(labels[0])
    for label in labels:
        if len(label) != leng:
            print("Padding operation erroneous")
            break
    print('There are {} unique symbols in the labels'.format(len(dic)))
    print('There are {} text sentences'.format(len(labels)))
    return paths, labels, dic


# Load and pad images with regard to the set of paths
def preprocess_images(paths, test = False):
    print("Loading images...")
    lines = './lines'
    images = []
    max_width = 0
    max_height = 0
    for path in paths:
        locs = path.split('-')
        path = '{}/{}/{}-{}/{}-{}-{}.png'.format(lines, locs[0], locs[0], locs[1], locs[0], locs[1], locs[2])
        image = io.imread(path, as_grey=True)
        height, width = image.shape
        if width > max_width: max_width = width
        if height > max_height: max_height = height
        images.append(image)
    size = len(images)
    print('There are {} handwritten lines'.format(size))
    print('Padding Images...')
    for i in range(size):
        height, width = images[i].shape
        images[i] = np.pad(array=images[i],
                           mode='constant',
                           pad_width=[(0, max_height-height), (0, max_width-width)],
                           constant_values=[0])
    return images


# Invoke this to build and train the model
def train():
    batch_size = 16
    epoch = 2
    paths, labels, dic = preprocess_labels()
    Xtrain, Ytrain, Xtest, Ytest, Xval, Yval = split_dataset_rand(paths, labels)
    model = full_model(num_of_class=len(dic))
    print("Compiling...")
    model.compile(optimizer=Adadelta(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    print("Training...")
    model.fit(x=Xtrain,
              y=Ytrain,
              verbose=1,
              batch_size=batch_size,
              epochs=epoch)
    model.evaluate(x=Xtest,
                   y=Ytest)
    model.save(filepath='./model.h5')


if __name__ == '__main__':
    #test = ['a01-000u-00', 'a01-000u-01', 'a01-000u-02', 'a01-000u-03', 'a01-000u-04']
    train()

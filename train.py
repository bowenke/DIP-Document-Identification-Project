import _collections as col
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, BatchNormalization, Dropout, \
                         Activation, Flatten, RepeatVector, TimeDistributed
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.preprocessing import sequence as seq
from keras.optimizers import Adadelta, RMSprop
from keras.losses import categorical_crossentropy
from skimage import io
# (342, 2270, 1)


def basic_model(num_of_class=81, input_shape=(342, 2270, 1), pad_len=95):
    model = Sequential()
    print("Input_shape: {}".format(input_shape))
    model.add(Conv2D(64,
                     input_shape=input_shape,
                     data_format='channels_last',
                     kernel_size=(3, 3),
                     activation='relu',
                     ))
    print("From Conv2D-1: {}".format(model.output_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("From MaxP-1: {}".format(model.output_shape))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu',
                     ))
    print("From Conv2D-2: {}".format(model.output_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("From MaxP-2: {}".format(model.output_shape))
    model.add(Dense(1, activation='relu'))
    print("Dense-1: {}".format(model.output_shape))
    model.add(Flatten())
    model.add(Reshape((21, 2264,)))
    print("Reshape: {}".format(model.output_shape))
    model.add(Bidirectional(LSTM(64, activation='relu'), merge_mode='concat'))
    print("Bidirectional: {}".format(model.output_shape))
    #model.add(Dense())
    model.add(RepeatVector(pad_len))
    print("RepeatVector: {}".format(model.output_shape))
    model.add(LSTM(num_of_class, activation='softmax', return_sequences=True))
    #model.add(TimeDistributed(Dense(num_of_class, activation='softmax')))
    #print("TimeDistributedDense: {}".format(model.output_shape))
    return model


def seq2seq_model(num_of_class=81, input_shape=(342, 2270, 1), pad_len=95):
    model = Sequential()
    print("Input_shape: {}".format(input_shape))
    model.add(Conv2D(64,
                     input_shape=input_shape,
                     data_format='channels_last',
                     kernel_size=(3, 3),
                     activation='relu',
                     ))
    print("From Conv2D-1: {}".format(model.output_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("From MaxP-1: {}".format(model.output_shape))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu',
                     ))
    print("From Conv2D-2: {}".format(model.output_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("From MaxP-2: {}".format(model.output_shape))
    model.add(Dense(1, activation='relu'))
    print("Dense-1: {}".format(model.output_shape))
    model.add(Flatten())
    model.add(Reshape((21, 2264,)))
    print("Reshape: {}".format(model.output_shape))

    return model


# Split data into training, testing and validation sets and load them
def split_dataset_rand(paths, labels, train_prop=0.85, test_prop=0.15, val=False):
    print("Preparing training, testing and validation datasets...")
    if len(paths) != len(labels):
        print("Fatal: data missing and not aligned")
    indices = np.arange(start=0, stop=len(paths), dtype=np.int64)
    np.random.shuffle(indices)
    leng = len(paths)
    train_size = int(train_prop*leng)
    test_size = int(test_prop*leng)
    training_indices = indices[:train_size]
    testing_indices = indices[train_size:train_size+test_size]
    validation_indices = indices[train_size+test_size:]
    X = preprocess_images(paths)
    Y = labels
    Xtrain = np.asarray([X[i] for i in training_indices])
    Ytrain = np.asarray([Y[i] for i in training_indices])
    Xtest = np.asarray([X[i] for i in testing_indices])
    Ytest = np.asarray([Y[i] for i in testing_indices])
    Xval = None
    Yval = None
    if leng > train_size+test_size and val:
        Xval = np.asarray([X[i] for i in validation_indices])
        Yval = np.asarray([labels[i] for i in validation_indices])
        print("Training Set: {} | Testing Set: {} | Validation Set: {}".format(train_size, test_size, leng-(train_size+test_size)))
        return Xtrain, Ytrain, Xtest, Ytest, Xval, Yval
    else:
        return Xtrain, Ytrain, Xtest, Ytest


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
def preprocess_images(paths):
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
        images[i] = images[i][:, :, np.newaxis]
    return images


# Invoke this to build and train the model
def train():

    paths, labels, dic = preprocess_labels()
    Xtrain, Ytrain, Xtest, Ytest = split_dataset_rand(paths, labels)

    batch_size = 4
    epoch = 2
    input_shape = Xtrain[0].shape
    pad_len = len(Ytrain[0])
    num_of_class = len(dic)


    model = basic_model(num_of_class=num_of_class, input_shape=input_shape, pad_len=pad_len)

    #model = basic_model()
    print("Compiling...")
    model.compile(optimizer=RMSprop(),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    print("Training...")
    print(Xtrain.shape)
    print(Ytrain.shape)
    model.fit(x=Xtrain,
              y=Ytrain,
              verbose=1,
              batch_size=1,
              epochs=epoch)
    # Manual Training
    #for x, y in zip(Xtrain, Ytrain):
    #    x = x[np.newaxis, :, :, :]
    #    y = y[np.newaxis, :, :]
    #    model.fit(x=x,
    #              y=y,
    #              verbose=1,
    #              batch_size=1,
    #              epochs=epoch)
    model.save(filepath='./model.h5')
    model.evaluate(x=Xtest,
                   y=Ytest)




if __name__ == '__main__':
    #test = ['a01-000u-00', 'a01-000u-01', 'a01-000u-02', 'a01-000u-03', 'a01-000u-04']
    train()

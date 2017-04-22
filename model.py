from keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, Flatten, BatchNormalization, Dropout
from keras.models import Sequential, Model


def data_streaming_from(directory):
    pass


def handwriting_identifier(num_of_class=-1):

    conv_net = Sequential()
    conv_net.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(None, None, 1)))
    conv_net.add(BatchNormalization())
    conv_net.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))
    conv_net.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    lstm = LSTM(256, activation='relu')

    enc_dec = Sequential()
    enc_dec.add(Flatten())
    enc_dec.add(Bidirectional(lstm))
    enc_dec.add(LSTM(256, activation='relu'))
    enc_dec.add(Dense(num_of_class, activation='softmax'))
    _model = Model(inputs=conv_net.input, outputs=enc_dec.output)
    return _model

if __name__ == '__main__':
    model = handwriting_identifier()

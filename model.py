from keras.layers import Conv2D, MaxPooling2D, LSTM, Bidirectional, Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.models import Sequential, Model


def handwriting_identifier(num_of_class=-1):

    conv_net = Sequential()

    # layer 1
    conv_net.add(Conv2D(256,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=2,
                        input_shape=(None, None, 1)))
    conv_net.add(BatchNormalization())

    # layer 2
    conv_net.add(Conv2D(256,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=2))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 3
    conv_net.add(Conv2D(64,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=2))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 4
    conv_net.add(Conv2D(64,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=2))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    # layer 5
    conv_net.add(Conv2D(32,
                        kernel_size=(3, 3),
                        activation='relu',
                        strides=2))
    conv_net.add(BatchNormalization())
    conv_net.add(MaxPooling2D(pool_size=(2, 2)))
    conv_net.add(Dropout(0.25))

    enc_dec = Sequential()
    # Encoder
    enc_dec.add(Flatten())
    lstm = LSTM(32, activation='relu')
    enc_dec.add(Bidirectional(lstm))

    # Decoder
    enc_dec.add(LSTM(256, activation='relu'))
    enc_dec.add(Dense(num_of_class))
    enc_dec.add(Activation('softmax'))

    _model = Model(inputs=conv_net.input, outputs=enc_dec(conv_net.output))
    return _model

if __name__ == '__main__':
    model = handwriting_identifier()

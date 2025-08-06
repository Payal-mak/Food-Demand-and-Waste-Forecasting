# model_architectures.py
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout

def build_lstm_multivariate(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inp)
    x = Dropout(0.3)(x)
    out1 = Dense(1, name='demand')(x)
    out2 = Dense(1, name='waste')(x)
    model = Model(inp, [out1, out2])
    return model

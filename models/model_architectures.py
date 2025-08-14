# models/model_architectures.py
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout

# Add lstm_units and dropout_rate as arguments with default values
def build_lstm_multivariate(input_shape, lstm_units=64, dropout_rate=0.3):
    inp = Input(shape=input_shape)
    # Use the arguments here
    x = LSTM(lstm_units, return_sequences=False)(inp)
    x = Dropout(dropout_rate)(x)
    out1 = Dense(1, name='demand')(x)
    out2 = Dense(1, name='waste')(x)
    model = Model(inp, [out1, out2])
    return model
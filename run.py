import tensorflow.keras as keras
from source.models import ADAE_LSTM, AdaeLstm

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()    
    gen, dis = ADAE_LSTM(x_train.shape[-1], x_train.shape[-1], x_train.shape[-2], hidden_size=32)
    x_train = x_train/x_train.max()
    adae_lstm = AdaeLstm(gen, dis)

    adae_lstm.compile(optimizer=[keras.optimizers.Adam(lr=1e-4, beta_1=0.5), keras.optimizers.Adam(lr=1e-4, beta_1=0.5)])#, metrics=[get_lr_metric])
    adae_lstm.fit(x_train[y_train != 0], batch_size=64, epochs=100, verbose=1, validation_split=0.2)
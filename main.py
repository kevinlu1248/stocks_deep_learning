import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim, xlim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import GRU, LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow_core.python.keras.layers import Flatten

dic = pd.read_csv('./data/ge.us.txt')
arr = dic.to_numpy()
trend = arr[:, 4]
dates = arr[:, 0].astype('datetime64', copy=False)
# trend = trend[12000:]
trend = trend[9000:]

data = trend - trend.mean()
data /= data.std()

lookback = 250
epochs = 20
step = 1
delay = 1
batch_size = 128
val_split = 0.15
test_split = 0.15
input_size = data.shape[0]  # 14058

do_plot_loss = 1
do_plot_history = 1

val_steps = (input_size * val_split - 1 - lookback) // batch_size


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            1,
                            lookback // step))
        targets = np.zeros((len(rows),))
        # print(rows)
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            # print(data[indices])
            samples[j] = [data[indices]]
            targets[j] = data[rows[j] + delay]
        yield samples, targets


# print(data.shape)

train_gen = generator(data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=int(input_size * (1 - val_split - test_split)),
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(data,
                    lookback=lookback,
                    delay=delay,
                    min_index=int(input_size * (1 - val_split - test_split)) + 1,
                    max_index=int(input_size * (1 - test_split)),
                    step=step,
                    batch_size=batch_size)
test_gen = generator(data,
                     lookback=lookback,
                     delay=delay,
                     min_index=int(input_size * (1 - test_split)) + 1,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

model_name = 'LSTM4_dropout'

model = Sequential([
    LSTM(32,
         input_shape=[None, lookback // step],
         dropout=0.1,
         recurrent_dropout=0.5,
         return_sequences=True),
    LSTM(32,
         input_shape=[None, lookback // step],
         dropout=0.1,
         recurrent_dropout=0.5),
    Dense(32, activation='relu'),
    Dense(1)
])

# # Covnet
# model = Sequential([
#     Conv1D(8, 5,
#            input_shape=[None, lookback // step],
#            activation='relu'),
#     MaxPooling1D(2),
#     Dense(1)
# ])

model.compile(optimizer='rmsprop',
              loss='mse')

model.summary()

history = model.fit(train_gen,
                    steps_per_epoch=500,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=val_steps)

model.save('models/{}.h5'.format(model_name))


def plot_loss():
    samples, targets = next(test_gen)
    predicted = model.predict(samples)
    predicted = predicted[:, 0]
    r = range(len(samples))
    real_line, = plt.plot(targets, label="Real")
    pred_line, = plt.plot(predicted, label="Prediction")
    plt.legend(handles=[real_line, pred_line])
    plt.savefig('figures/{}.png'.format(model_name))
    plt.show()


def plot_history():
    loss, val_loss = history.history['loss'], history.history['val_loss']
    loss_line, = plt.plot(loss, label="Loss")
    val_line, = plt.plot(val_loss, label="Validation_loss")
    plt.legend(handles=[loss_line, val_line])
    ylim(0, 0.3)
    plt.savefig('figures/{}_loss.png'.format(model_name))
    plt.show()


def plot_data():
    plt.plot(data)
    plt.show()


if do_plot_loss:
    plot_loss()

if do_plot_history:
    plot_history()

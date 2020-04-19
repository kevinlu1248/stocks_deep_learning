import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

dic = pd.read_csv('./data/ge.us.txt')
arr = dic.to_numpy()
trend = arr[:, 4]
dates = arr[:, 0].astype('datetime64', copy=False)

# normalizing
trend -= trend.mean()
trend /= trend.std()

lv_trend = trend[1:]
trend = trend[:-1]
diff = lv_trend - trend
print((diff**2).mean())
# baseline is mse = 0.000553396536661067

# data = trend - trend.mean()
# data /= data.std()
# print(data)

# plt.plot(dates[:-1], trend, 'b-', dates[:-1], lv_trend, 'r-')
# plt.plot(dates[1:], diff)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dic = pd.read_csv('./data/ge.us.txt')
# print(dic)a
arr = dic.to_numpy()
trend = arr[:, 4]
# dates = arr[:, 0]
dates = arr[:, 0].astype('datetime64', copy=False)
# dates = np.datetime_as_string(arr[:, 0])
plt.plot(dates, trend)
plt.show()

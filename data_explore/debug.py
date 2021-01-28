import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["FangSong"]
plt.rcParams["axes.unicode_minus"] = False
import warnings

warnings.filterwarnings("ignore")

X = pd.DataFrame(np.array([[ 10, 0, True], [10, 0, True], [ 10, 0, True],
                           [ 70, 0, False], [ 10, 0, True], [ 50, 0, True],
                           [ 10, 9, True]]),
                 columns=[ 'Type', 'zero', 'isor'])
y = np.array([0, 0, 0, 0, 0, 0, 1])
# 测试各种编码的方法
# 1.目标编码，首选尝试
print(X)
X.index = [i for i in range(17,10,-1)]
print(X)

# 2.留一法编码
# 3.catboost目标编码
# 4.WOEEncoder证据权重编码
# 5.CountEncoder频率编码

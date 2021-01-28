# utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

np.random.seed(0)
datafile = "F:/比赛相关项目/软件服务外包大赛/A04/" + 'train_20210120_date_split.csv'  # 航空原始数据,第一行为属性标签

data_train = pd.read_csv(datafile)
data_target = data_train['emd_lable2'].to_numpy()
data_train_feature = data_train.drop(['emd_lable2'], axis=1)
print('特征列', data_train_feature.shape, '目标列', data_target.shape)

feature_list = list(data_train_feature.columns.array)
discrete_list = ['pax_name', 'pax_passport', 'seg_route_from', 'seg_route_to', 'seg_flight', 'seg_cabin',
                 'seg_dep_time',
                 'emd_lable', 'gender', 'age', 'birth_date', 'residence_country', 'nation_name', 'city_name',
                 'province_name', 'marital_stat', 'ffp_nbr', 'member_level', 'often_city', 'enroll_chnl',
                 'pref_aircraft_m3_1', 'pref_aircraft_m3_2', 'pref_aircraft_m3_3', 'pref_aircraft_m3_4',
                 'pref_aircraft_m3_5', 'pref_aircraft_m6_1', 'pref_aircraft_m6_2', 'pref_aircraft_m6_3',
                 'pref_aircraft_m6_4', 'pref_aircraft_m6_5', 'pref_aircraft_y1_1', 'pref_aircraft_y1_2',
                 'pref_aircraft_y1_3', 'pref_aircraft_y1_4', 'pref_aircraft_y1_5', 'pref_aircraft_y2_1',
                 'pref_aircraft_y2_2', 'pref_aircraft_y2_3', 'pref_aircraft_y2_4', 'pref_aircraft_y2_5',
                 'pref_aircraft_y3_1', 'pref_aircraft_y3_2', 'pref_aircraft_y3_3', 'pref_aircraft_y3_4',
                 'pref_aircraft_y3_5', 'pref_orig_m3_1', 'pref_orig_m3_2', 'pref_orig_m3_3', 'pref_orig_m3_4',
                 'pref_orig_m3_5', 'pref_orig_m6_1', 'pref_orig_m6_2', 'pref_orig_m6_3', 'pref_orig_m6_4',
                 'pref_orig_m6_5', 'pref_orig_y1_1', 'pref_orig_y1_2', 'pref_orig_y1_3', 'pref_orig_y1_4',
                 'pref_orig_y1_5', 'pref_orig_y2_1', 'pref_orig_y2_2', 'pref_orig_y2_3', 'pref_orig_y2_4',
                 'pref_orig_y2_5', 'pref_orig_y3_1', 'pref_orig_y3_2', 'pref_orig_y3_3', 'pref_orig_y3_4',
                 'pref_orig_y3_5', 'pref_line_m3_1', 'pref_line_m3_2', 'pref_line_m3_3', 'pref_line_m3_4',
                 'pref_line_m3_5', 'pref_line_m6_1', 'pref_line_m6_2', 'pref_line_m6_3', 'pref_line_m6_4',
                 'pref_line_m6_5', 'pref_line_y1_1', 'pref_line_y1_2', 'pref_line_y1_3', 'pref_line_y1_4',
                 'pref_line_y1_5', 'pref_line_y2_1', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y2_4',
                 'pref_line_y2_5', 'pref_line_y3_1', 'pref_line_y3_2', 'pref_line_y3_3', 'pref_line_y3_4',
                 'pref_line_y3_5', 'pref_city_m3_1', 'pref_city_m3_2', 'pref_city_m3_3', 'pref_city_m3_4',
                 'pref_city_m3_5', 'pref_city_m6_1', 'pref_city_m6_2', 'pref_city_m6_3', 'pref_city_m6_4',
                 'pref_city_m6_5', 'pref_city_y1_1', 'pref_city_y1_2', 'pref_city_y1_3', 'pref_city_y1_4',
                 'pref_city_y1_5', 'pref_city_y2_1', 'pref_city_y2_2', 'pref_city_y2_3', 'pref_city_y2_4',
                 'pref_city_y2_5', 'pref_city_y3_1', 'pref_city_y3_2', 'pref_city_y3_3', 'pref_city_y3_4',
                 'pref_city_y3_5', 'pref_month_m3_1', 'pref_month_m3_2', 'pref_month_m3_3', 'pref_month_m3_4',
                 'pref_month_m3_5', 'pref_month_m6_1', 'pref_month_m6_2', 'pref_month_m6_3', 'pref_month_m6_4',
                 'pref_month_m6_5', 'pref_month_y1_1', 'pref_month_y1_2', 'pref_month_y1_3', 'pref_month_y1_4',
                 'pref_month_y1_5', 'pref_month_y2_1', 'pref_month_y2_2', 'pref_month_y2_3', 'pref_month_y2_4',
                 'pref_month_y2_5', 'pref_month_y3_1', 'pref_month_y3_2', 'pref_month_y3_3', 'pref_month_y3_4',
                 'pref_month_y3_5', 'recent_flt_day', 'pit_add_chnl_m3', 'pit_add_chnl_m6', 'pit_add_chnl_y1',
                 'pit_add_chnl_y2', 'pit_add_chnl_y3', 'pref_orig_city_m3', 'pref_orig_city_m6', 'pref_orig_city_y1',
                 'pref_orig_city_y2', 'pref_orig_city_y3', 'pref_dest_city_m3', 'pref_dest_city_m6',
                 'pref_dest_city_y1',
                 'pref_dest_city_y2', 'pref_dest_city_y3'
    , 'seg_dep_time_month', 'seg_dep_time_year',
                 'seg_dep_time_is_workday',
                 'seg_dep_time_is_holiday', 'seg_dep_time_is_in_lieu'
                 ]
continue_list = list(set(feature_list) - set(discrete_list))
print('特征列表长度为{0},离散特征长度{1},连续特征长度{2}'.format(len(feature_list), len(discrete_list), len(continue_list)))


# 划分数据集
def get_train_test(data_train_feature, data_target):
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, test_index in kfold.split(data_train_feature, data_target):
        return train_index, test_index


train_index, test_index = get_train_test(data_train_feature, data_target)
x_train = data_train_feature.loc[train_index]
x_test = data_train_feature.loc[test_index]
y_train = data_target[train_index]
y_test = data_target[test_index]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(np.mean(y_train), np.mean(y_test))  # 比较训练集与测试集的分布情况

import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler


def minmax_target(x_train, x_test, y_train):
    # 记录解决方案：采用离散特征与连续特征分离的方法
    # 保留minmax和encoder，对测试集分离后分别进行编码和归一化，再整合起来
    #这条语句就是需要修改的编码器
    encoder = ce.TargetEncoder(cols=discrete_list, drop_invariant=False).fit(x_train, y_train)


    x_train = encoder.transform(x_train)  # 基于训练集得到编码器

    minmax = MinMaxScaler()
    minmax.fit(x_train[continue_list])  # 基于训练集得到归一化
    x_train[continue_list] = minmax.transform(x_train[continue_list])
    # print(x_train.shape, continue_x_train.shape)

    x_test = encoder.transform(x_test)
    x_test[continue_list] = minmax.transform(x_test[continue_list])  # 连续值处理
    # print(x_test.shape, continue_x_test.shape)

    return x_train, x_test


# x_train,x_test列表前面的标号index，影响了之后单独对连续值特征归一化！将index设置为从0开始
x_train.index = [i for i in range(x_train.shape[0])]
x_test.index = [i for i in range(x_test.shape[0])]
x_train, x_test = minmax_target(x_train, x_test, y_train)
# encoder = ce.TargetEncoder(cols=discrete_list, drop_invariant=False).fit(data_train_feature, data_target)
# data_train_feature = encoder.transform(data_train_feature)
# print('重新编码后的特征长度{0},删除了{1}个方差为0的特征,他们是{2}'.format(data_train_feature.shape,
#                                                    len(feature_list) - data_train_feature.shape[1],
#                                                    encoder.drop_cols))
# minmax = MinMaxScaler()
# data_train_feature = minmax.fit_transform(data_train_feature)

from sklearn.tree import DecisionTreeClassifier

dTree = DecisionTreeClassifier()
dTree.fit(x_train, y_train)
y_pred = dTree.predict(x_test)


def evaluation(y_pred, y_true):
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    print("balanced_accuracy_score=", balanced_accuracy_score(y_pred=y_pred, y_true=y_true))
    print("f1=", f1_score(y_pred=y_pred, y_true=y_true))
    print("precision_score=", precision_score(y_pred=y_pred, y_true=y_true))
    print("recall_score=", recall_score(y_pred=y_pred, y_true=y_true))


evaluation(y_pred=y_pred, y_true=y_test)

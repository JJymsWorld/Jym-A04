import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']

datafile = "D:/2021软件服务外包比赛/data_split/" + 'train_pre_1000.xlsx'  # 航空原始数据,第一行为属性标签
resultfile = "D:/2021软件服务外包比赛/result/" + 'pre_1000_result_20210101.xls'  # 数据探索结果表

data_train = pd.read_excel(datafile)
df_target = data_train['emd_lable2']
data_train_feature = data_train.drop(['emd_lable2'], axis=1)

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
                 ]+['pref_month_m3_1', 'pref_month_m3_2', 'pref_month_m3_3', 'pref_month_m3_4', 'pref_month_m3_5',
                 'pref_month_m6_1', 'pref_month_m6_2', 'pref_month_m6_3', 'pref_month_m6_4', 'pref_month_m6_5',
                 'pref_month_y1_1', 'pref_month_y1_2', 'pref_month_y1_3', 'pref_month_y1_4', 'pref_month_y1_5',
                 'pref_month_y2_1', 'pref_month_y2_2', 'pref_month_y2_3', 'pref_month_y2_4', 'pref_month_y2_5',
                 'pref_month_y3_1', 'pref_month_y3_2', 'pref_month_y3_3', 'pref_month_y3_4', 'pref_month_y3_5',
                 ]
continue_list = list(set(feature_list) - set(discrete_list))

# 绘制每个特征与类标签的关系图

# f = open("explore.txt", 'w+')
# for n,i in enumerate(feature_list):
#     Cabin_cat_num = data_train_feature[i].value_counts().index.shape[0]
#     print('{0},{1}特征的类型数量为{2}'.format(n+1,i,Cabin_cat_num),file=f)
# print('共有{0}个离散化特征'.format(len(discrete_list)))
# for k in range(20):
#     fig, axes = plt.subplots(2, 4,figsize=(20,10))
#     for i in range(8):
#         row = int(i / 4)
#         hei = i % 4
#         sns.countplot(x=discrete_list[i], hue='emd_lable2', data=data_train, ax=axes[row, hei])
#         # axes[row, hei].set_title('{0}特征分析'.format(discrete_list[i]))
#     fig.suptitle("离散属性特征分析", size=20, y=1.1)
#     plt.savefig("D:/2021软件服务外包比赛/images1/"+str(k)+".jpg")
#     discrete_list = discrete_list[8:]


#删除的和无关的特征

#对离散取值非常多的特征进行处理，1.城市省份归一化2.时间日期连续化

# 特征之间的相关性可以画一下，采用各种相关系数，如mic，传统的corr等
# print(len(remove_list),remove_list)
# for k in range(62):
#     f, ax = plt.subplots(2, 4, figsize=(20, 10))
#     for i in range(8):
#         row = int(i / 4)
#         hei = i % 4
#         sns.kdeplot(data_train.loc[(data_train['emd_lable2'] == 0), remove_list[i]], color='gray', shade=True,
#                     label='lable=0',ax=ax[row, hei])
#         sns.kdeplot(data_train.loc[(data_train['emd_lable2'] == 1), remove_list[i]], color='g', shade=True,
#                     label='label=1',ax=ax[row, hei])
#         ax[row, hei].set_title('{0}特征分析'.format(remove_list[i]))
#     f.suptitle("连续属性特征分析", size=20, y=1.1)
#     plt.savefig("D:/2021软件服务外包比赛/images/" + str(k) + ".jpg")
#     remove_list = remove_list[8:]

# 剩余需要进行特征选择与模型建立的特征
# 离散值特征的编码预处理


# 连续值特征的预处理
# 1.是否需要离散化






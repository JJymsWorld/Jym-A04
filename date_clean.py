# [A04付费旅客]
import pandas as pd
from datetime import datetime
from chinese_calendar import is_workday,is_holiday,is_in_lieu#是否工作日，休息日，法定节假日


datafile = "D:/2021软件服务外包比赛/result/" + 'train_20210120_date_split.csv'  # 航空原始数据,第一行为属性标签
resultfile = "D:/2021软件服务外包比赛/result/" + 'train_20210120_date_split.csv'  # 数据探索结果表


print(is_workday(datetime.now()))
print(is_holiday(datetime.now()))
print(is_in_lieu(datetime.now()))
data_train = pd.read_csv(datafile)
print(data_train['seg_dep_time'][0])
print(datetime.strptime(data_train['seg_dep_time'][0],'%Y/%m/%d %H:%M'))
# data_train['seg_dep_time_month'] = [data_train['seg_dep_time'][i].month for i in
#                                     range(len(data_train))]
# data_train['seg_dep_time_year'] = [data_train['seg_dep_time'][i].year for i in
#                                    range(len(data_train))]
data_train['seg_dep_time_is_workday'] = [is_workday(datetime.strptime(data_train['seg_dep_time'][i],'%Y/%m/%d %H:%M'))
                                         for i in
                                         range(len(data_train))]
data_train['seg_dep_time_is_holiday'] = [is_holiday(datetime.strptime(data_train['seg_dep_time'][i],'%Y/%m/%d %H:%M'))
                                         for i in
                                         range(len(data_train))]
data_train['seg_dep_time_is_in_lieu'] = [is_in_lieu(datetime.strptime(data_train['seg_dep_time'][i],'%Y/%m/%d %H:%M'))
                                         for i in
                                         range(len(data_train))]

# print(data_train['seg_dep_time_month'])
# print(data_train['seg_dep_time_year'])
# print(data_train['seg_dep_time_is_workday'])
#
data_train.to_csv(resultfile,encoding='utf-8')
# data_target = data_train['emd_lable2'].to_numpy()
# data_train_feature = data_train.drop(['emd_lable2'], axis=1)
# print('特征列', data_train_feature.shape, '目标列', data_target.shape)
#
# feature_list = list(data_train_feature.columns.array)
# discrete_list = ['pax_name', 'pax_passport', 'seg_route_from', 'seg_route_to', 'seg_flight', 'seg_cabin',
#                  'seg_dep_time',
#                  'emd_lable', 'gender', 'age', 'birth_date', 'residence_country', 'nation_name', 'city_name',
#                  'province_name', 'marital_stat', 'ffp_nbr', 'member_level', 'often_city', 'enroll_chnl',
#                  'pref_aircraft_m3_1', 'pref_aircraft_m3_2', 'pref_aircraft_m3_3', 'pref_aircraft_m3_4',
#                  'pref_aircraft_m3_5', 'pref_aircraft_m6_1', 'pref_aircraft_m6_2', 'pref_aircraft_m6_3',
#                  'pref_aircraft_m6_4', 'pref_aircraft_m6_5', 'pref_aircraft_y1_1', 'pref_aircraft_y1_2',
#                  'pref_aircraft_y1_3', 'pref_aircraft_y1_4', 'pref_aircraft_y1_5', 'pref_aircraft_y2_1',
#                  'pref_aircraft_y2_2', 'pref_aircraft_y2_3', 'pref_aircraft_y2_4', 'pref_aircraft_y2_5',
#                  'pref_aircraft_y3_1', 'pref_aircraft_y3_2', 'pref_aircraft_y3_3', 'pref_aircraft_y3_4',
#                  'pref_aircraft_y3_5', 'pref_orig_m3_1', 'pref_orig_m3_2', 'pref_orig_m3_3', 'pref_orig_m3_4',
#                  'pref_orig_m3_5', 'pref_orig_m6_1', 'pref_orig_m6_2', 'pref_orig_m6_3', 'pref_orig_m6_4',
#                  'pref_orig_m6_5', 'pref_orig_y1_1', 'pref_orig_y1_2', 'pref_orig_y1_3', 'pref_orig_y1_4',
#                  'pref_orig_y1_5', 'pref_orig_y2_1', 'pref_orig_y2_2', 'pref_orig_y2_3', 'pref_orig_y2_4',
#                  'pref_orig_y2_5', 'pref_orig_y3_1', 'pref_orig_y3_2', 'pref_orig_y3_3', 'pref_orig_y3_4',
#                  'pref_orig_y3_5', 'pref_line_m3_1', 'pref_line_m3_2', 'pref_line_m3_3', 'pref_line_m3_4',
#                  'pref_line_m3_5', 'pref_line_m6_1', 'pref_line_m6_2', 'pref_line_m6_3', 'pref_line_m6_4',
#                  'pref_line_m6_5', 'pref_line_y1_1', 'pref_line_y1_2', 'pref_line_y1_3', 'pref_line_y1_4',
#                  'pref_line_y1_5', 'pref_line_y2_1', 'pref_line_y2_2', 'pref_line_y2_3', 'pref_line_y2_4',
#                  'pref_line_y2_5', 'pref_line_y3_1', 'pref_line_y3_2', 'pref_line_y3_3', 'pref_line_y3_4',
#                  'pref_line_y3_5', 'pref_city_m3_1', 'pref_city_m3_2', 'pref_city_m3_3', 'pref_city_m3_4',
#                  'pref_city_m3_5', 'pref_city_m6_1', 'pref_city_m6_2', 'pref_city_m6_3', 'pref_city_m6_4',
#                  'pref_city_m6_5', 'pref_city_y1_1', 'pref_city_y1_2', 'pref_city_y1_3', 'pref_city_y1_4',
#                  'pref_city_y1_5', 'pref_city_y2_1', 'pref_city_y2_2', 'pref_city_y2_3', 'pref_city_y2_4',
#                  'pref_city_y2_5', 'pref_city_y3_1', 'pref_city_y3_2', 'pref_city_y3_3', 'pref_city_y3_4',
#                  'pref_city_y3_5', 'pref_month_m3_1', 'pref_month_m3_2', 'pref_month_m3_3', 'pref_month_m3_4',
#                  'pref_month_m3_5', 'pref_month_m6_1', 'pref_month_m6_2', 'pref_month_m6_3', 'pref_month_m6_4',
#                  'pref_month_m6_5', 'pref_month_y1_1', 'pref_month_y1_2', 'pref_month_y1_3', 'pref_month_y1_4',
#                  'pref_month_y1_5', 'pref_month_y2_1', 'pref_month_y2_2', 'pref_month_y2_3', 'pref_month_y2_4',
#                  'pref_month_y2_5', 'pref_month_y3_1', 'pref_month_y3_2', 'pref_month_y3_3', 'pref_month_y3_4',
#                  'pref_month_y3_5', 'recent_flt_day', 'pit_add_chnl_m3', 'pit_add_chnl_m6', 'pit_add_chnl_y1',
#                  'pit_add_chnl_y2', 'pit_add_chnl_y3', 'pref_orig_city_m3', 'pref_orig_city_m6', 'pref_orig_city_y1',
#                  'pref_orig_city_y2', 'pref_orig_city_y3', 'pref_dest_city_m3', 'pref_dest_city_m6',
#                  'pref_dest_city_y1',
#                  'pref_dest_city_y2', 'pref_dest_city_y3'
#                  ]
# continue_list = list(set(feature_list) - set(discrete_list))
# print('特征列表长度为{0},离散特征长度{1},连续特征长度{2}'.format(len(feature_list), len(discrete_list), len(continue_list)))

# 节假日，工作日、非工作日，休息日的出行统计次数计算（数据补充，明显存在统计错误）？

# encoding:utf-8
import copy
from operator import attrgetter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
from minepy import MINE
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

from passenger_identify.feature_selection.Fitness import Data, Test_Data
from passenger_identify.feature_selection.Partical import Particle


def getTrainTest(feature, target):
    # 这里可以根据数据集大小，按顺序抽取样本为测试集、训练集
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # 按顺序获取测试集训练集
    train_test = []
    for train_index, test_index in kf.split(feature, target):
        temp = [train_index, test_index]
        train_test.append(temp)
    return train_test


def swap(solution, purpose, arr):
    '''定义交换操作SO'''
    temp = solution[arr[0]]
    solution[arr[0]] = purpose[arr[1]]
    purpose[arr[1]] = temp


def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def generateSS(solution, purpose):
    '''核心学习过程：生成学习队列，现采用集合论方式生成，对重复情况有一定取舍'''
    ss = []
    solutionSet = set(solution)
    purposeSet = set(purpose)
    # 求对应的交集
    remove = solutionSet & purposeSet
    # 去除交集，获取差异特征序列
    solutionSet = solutionSet - remove
    purposeSet = purposeSet - remove
    # 针对差异特征较短的集合，进行替换产生so[a,b]
    length = len(solutionSet) if (len(solutionSet) < len(purposeSet)) else len(purposeSet)
    if length == 0: return ss
    solutionSet = list(solutionSet)
    purposeSet = list(purposeSet)
    for i in range(length):
        a = solution.index(solutionSet[i])
        b = purpose.index(purposeSet[i])
        ss.append([a, b])
    return ss


class PSO:
    def __init__(self, iterations, obj, alpha, beta):
        self.iterations = iterations  # max of iterations
        self.particles = []  # list of particles
        self.obj = obj
        self.alpha = alpha
        self.beta = beta
        self.all_feature_size = obj.getDimension()
        self.size_population = 50  # size population
        self.mic_feature_list = self.feature_filter()
        self.choose = 1
        # creates the particles and initialization
        self.solutions = self.RWS(self.mic_feature_list, self.get_subset())
        self.top_features = self.get_top_features()
        for solution in self.solutions:
            # creates a new particle
            particle = Particle(solution=solution, cost=obj.getTrainAccuracy(features=solution))
            # add the particle
            self.particles.append(particle)
        # update gbest
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        self.best = copy.copy(self.gbest)

    def get_subset(self):
        # 初始的特征子集长度影响非常大
        return 5

    def RWS(self, mic_feature_list, size):
        # 权重与对应的选择长度
        constant = [0, 1 / 10, 3 / 10, 6 / 10, 1]
        sub_mic_feature_list = mic_feature_list[:int(constant[self.choose] * len(mic_feature_list))]
        # 先进行随机选择，看起来比较方便
        P_MIC = []
        for mic, feature in sub_mic_feature_list:
            P_MIC.append(mic)
        P_MIC_SUM = np.sum(P_MIC)
        P_MIC = []
        P_MIC_feature = []
        for mic, feature in sub_mic_feature_list:
            P_MIC.append(mic / P_MIC_SUM)
            P_MIC_feature.append(feature)
        for i in range(len(P_MIC)):
            if i != 0:
                P_MIC[i] = P_MIC[i] + P_MIC[i - 1]
        solutions = []
        for par in range(self.size_population):
            solution = []
            for i in range(size):
                rand = np.random.rand()
                for j in range(len(P_MIC)):
                    if rand < P_MIC[j]:
                        solution.append(P_MIC_feature[j])
                        break
            solutions.append(solution)
        return solutions

    def get_top_features(self):
        sub_feature_mic_list = self.mic_feature_list[:int(0.02 * self.all_feature_size)]
        top_feature = []
        for mic, feature in sub_feature_mic_list:
            top_feature.append(feature)
        return top_feature

    def feature_filter(self):
        """基于MIC相关系数，进行特征排序筛选， 产生一个排序后的字典{MIC,feature index}"""
        feature_mic = {}
        mine = MINE(alpha=0.6, c=15)
        for i in range(self.all_feature_size):
            mine.compute_score(self.obj._X_train[:, i], self.obj._Y_train)
            feature_mic[i] = mine.mic()
        feature_mic_list = []
        for feature, mic in feature_mic.items():
            # if mic > 0.3:
            feature_mic_list.append((mic, feature))
        # print("经过粗筛选的特征共有",len(feature_mic_list),"被删除的特征数量",self.obj.getDimension()-len(feature_mic_list))
        feature_mic_list.sort(reverse=True)
        # name = ['feature', 'mic_index']
        # pd.DataFrame(columns=name, data=feature_mic_list).to_csv("D:/2021软件服务外包比赛/result/feature_mic.csv")
        return feature_mic_list

    def showParticle(self, t):
        print('迭代次数为{2}   gbest  length={0}   fitness={1}'.format(len(self.gbest.getPBest()), self.gbest.getCostPBest(),
                                                                  t))

    def run(self):
        count = 0
        t = 0
        while t < self.iterations:
            # update each particle's solution
            for particle in self.particles:
                solution_gbest = self.gbest.getPBest()[:]  # gets solution of the gbest
                solution_pbest = particle.getPBest()[:]  # copy of the pbest solution
                solution_particle = particle.getCurrentSolution()[:]
                # if np.random.rand() < 0.1:
                #     # start = particle.getCostPBest()
                #     particle.setCurrentSolution(self.mutation(solution_particle))
                #     # end = self.obj.getTrainAccuracy(particle.getCurrentSolution())
                #     # print('start local search')
                #     continue
                ss_pbest = generateSS(solution_particle, solution_pbest)
                for so in ss_pbest:
                    alpha = np.random.random()
                    if alpha < self.alpha:
                        swap(solution_particle, solution_pbest, so)
                ss_gbest = generateSS(solution_particle, solution_gbest)
                for so in ss_gbest:
                    beta = np.random.random()
                    if beta < self.beta:
                        swap(solution_particle, solution_gbest, so)
                # update current_solution
                particle.setCurrentSolution(solution_particle)

            # update pbest,gbest
            for particle in self.particles:
                if particle.getPBest() != particle.getCurrentSolution():
                    particle.setCostCurrentSolution(
                        self.obj.getTrainAccuracy(particle.getCurrentSolution()))
                    if particle.getCostCurrentSolution() > particle.getCostPBest():
                        particle.setPBest(particle.getCurrentSolution())
                        particle.setCostPBest(particle.getCostCurrentSolution())
            self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))

            # resize the feature_subset
            if self.gbest.getCostPBest() > self.best.getCostPBest():
                self.best = copy.copy(self.gbest)
                count = 1
                print("best更新成功！！！！！！！！！！！！！！！")
            else:
                count += 1
            if count == 5:
                self.re_size_ent()
            elif count == 15:  # 当然，陷入停滞了就完蛋了啥都没有
                t = t - 10
                if self.choose == 4:
                    break
                self.back()
            self.showParticle(t)
            t = t + 1

    def generateSS_DLS(self, solution, purpose):
        '''核心学习过程：生成学习队列，现采用集合论方式生成，对重复情况有一定取舍'''
        ss = []
        solutionSet = set(solution)
        purposeSet = set(purpose)
        # 求对应的交集
        remove = solutionSet & purposeSet
        # 去除交集，获取差异特征序列
        solutionSet = solutionSet - remove
        purposeSet = purposeSet - remove
        # 针对差异特征较短的集合，进行替换产生so[a,b]
        length = len(solutionSet) if (len(solutionSet) < len(purposeSet)) else len(purposeSet)
        if length == 0: return ss
        solutionSet = list(solutionSet)
        purposeSet = list(purposeSet)
        # 引入DLS，存储变化过程的中间变量粒子
        particle_temp = Particle(solution=solution[:],
                                 cost=self.obj.getTrainAccuracy(solution))
        purpose_temp = purpose[:]
        for i in range(length):
            a = solution.index(solutionSet[i])
            b = purpose.index(purposeSet[i])
            # 交换学习
            swap(particle_temp.getCurrentSolution(), purpose_temp, [a, b])
            acc_temp = self.obj.getTrainAccuracy(features=particle_temp.getCurrentSolution())
            if acc_temp > particle_temp.getCostCurrentSolution():
                # 更新准确度，确保只增不减，保留次交换行为
                particle_temp.setCostCurrentSolution(acc_temp)
                ss.append([a, b])
            else:  # 撤销之前的交换行为
                swap(particle_temp.getCurrentSolution(), purpose_temp, [a, b])
        return ss

    def re_size_ent(self):
        '''对于不是gbest的所有粒子重置，gbest保留历史信息'''
        size = int(2 * calc_ent(self.obj._Y_train) / self.gbest.getCostPBest() + 3)
        self.solutions = self.RWS(self.mic_feature_list, size)
        self.old_particles = copy.copy(self.particles)
        for i in range(len(self.particles)):
            solution = self.particles[i].getCurrentSolution() + self.solutions[i][:size]
            self.particles[i] = Particle(solution=solution,
                                         cost=obj.getTrainAccuracy(features=solution))

        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        print('第{0}个区间进行更新'.format(self.choose))
        if self.gbest.getCostPBest() > self.best.getCostPBest():
            self.best = copy.copy(self.gbest)
            print("best更新成功！！！！！！！！！！！！！！！")

    def back(self):
        print("撤回原来的操作，更新走向下一个区块，返还之前的迭代次数,重新进行count次数计算")

        self.choose = self.choose + 1
        self.particles = self.old_particles
        self.gbest = max(self.particles, key=attrgetter('cost_pbest_solution'))
        self.re_size_ent()


if __name__ == "__main__":
    datafile = "D:/2021软件服务外包比赛/result/" + 'train_20210120_date_split.csv'  # 航空原始数据,第一行为属性标签

    data_train = pd.read_csv(datafile)
    data_target = data_train['emd_lable2'].to_numpy()
    data_train_feature = data_train.drop(['emd_lable2'], axis=1)
    print('特征列', data_train_feature.shape, '目标列', data_target.shape)

    feature_list = list(data_train_feature.columns.array)
    discrete_list = [  # 会员编号,常飞月份为离散特征
        'pax_name', 'pax_passport', 'seg_route_from', 'seg_route_to', 'seg_flight', 'seg_cabin',
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
        'pit_add_chnl_y2', 'pit_add_chnl_y3', 'pref_orig_city_m3', 'pref_orig_city_m6',
        'pref_orig_city_y1',
        'pref_orig_city_y2', 'pref_orig_city_y3', 'pref_dest_city_m3', 'pref_dest_city_m6',
        'pref_dest_city_y1',
        'pref_dest_city_y2', 'pref_dest_city_y3'
        , 'seg_dep_time_month', 'seg_dep_time_year', 'seg_dep_time_is_workday'
    ]
    continue_list = list(set(feature_list) - set(discrete_list))
    print('特征列表长度为{0},离散特征长度{1},连续特征长度{2}'.format(len(feature_list), len(discrete_list), len(continue_list)))

    # 将离散数据进行target_encoding
    encoder = ce.TargetEncoder(cols=discrete_list, drop_invariant=False).fit(data_train_feature,
                                                                             data_target)
    data_train_feature = encoder.transform(data_train_feature).to_numpy()
    train_test_split = getTrainTest(data_train_feature, data_target)

    for train_index, test_index in train_test_split:
        np.random.seed(0)
        obj = Data(data_train_feature, data_target, train_index)
        obj2 = Test_Data(data_train_feature, data_target, train_index, test_index)
        pso = PSO(iterations=100, obj=obj, beta=0.2, alpha=0.4)
        pso.run()
        print('得到的特征子集序列为', pso.best.getPBest())
        print('特征子集长度为', len(set(pso.best.getPBest())))
        print('训练集准确率(适应度)为', pso.best.getCostPBest())
        print('得到的测试集准确率为', obj2.getTestAccuracy(pso.best.getPBest()))
        print('得到的测试集F1值为', obj2.getTestF1(pso.best.getPBest()))

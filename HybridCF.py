import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")

now_phase = 5
train_path = 'underexpose_train'
test_path = 'underexpose_test'

flag_append = False
flag_test = False
recall_num = 500
topk = 50
nrows = None
alpha = 0.7

submit_all = pd.DataFrame()
click_all = pd.DataFrame()


def interfere():
    import sys
    sys.exit(0)


def HybridCF(click_all, dict_label, k=100):
    """
    :param click_all: 点击数据
    :param dict_label: 标签
    :param k: 召回个数
    :return: topk结果
    """
    from collections import Counter
    group_by_col, agg_col = 'user_id', 'item_id'
    data_item = click_all.groupby(['user_id'])[['item_id', 'time']].agg(
        {'item_id': lambda x: ','.join(list(x)), 'time': lambda x: ','.join(list(x))}
    ).reset_index()
    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    stat_cnt = Counter(list(click_all['item_id']))
    stat_length = np.mean([len(item_txt.split(',')) for item_txt in data_item['item_id']])
    print("数据格式：", data_item.shape)
    # print(len(data_.iloc[0]['item_id'].split(',')))
    # print(len(data_.iloc[0]['time'].split(',')))
    # print(data_item.loc['10'])
    # print(data_item)
    # interfere()

    item_matrix_association_rules = {}
    user_matrix_association_rules = {}

    print("------- association rules 生成 ---------")

    for i, row in tqdm(data_item.iterrows()):
        list_item_id = row['item_id'].split(',')
        list_time = row['time'].split(',')
        len_list_item = len(list_item_id)

        for i, (item_i, time_i) in enumerate(zip(list_item_id, list_time)):
            for j, (item_j, time_j) in enumerate(zip(list_item_id, list_time)):

                t = np.abs(float(time_i) - float(time_j))
                d = np.abs(i - j)

                if i < j:
                    if item_i not in item_matrix_association_rules:
                        item_matrix_association_rules[item_i] = {}
                    if item_j not in item_matrix_association_rules[item_i]:
                        item_matrix_association_rules[item_i][item_j] = 0

                    item_matrix_association_rules[item_i][item_j] += 1 * 0.7 * (0.8 ** (d - 1)) * \
                                                                     (1 - t * 10000) / np.log(
                        1 + len_list_item)

                if i > j:
                    if item_i not in item_matrix_association_rules:
                        item_matrix_association_rules[item_i] = {}
                    if item_j not in item_matrix_association_rules[item_i]:
                        item_matrix_association_rules[item_i][item_j] = 0

                    item_matrix_association_rules[item_i][item_j] += 1 * 1.0 * (0.8 ** (d - 1)) * (
                            1 - t * 10000) / np.log(
                        1 + len_list_item)

    assert len(item_matrix_association_rules.keys()) == len(set(click_all['item_id']))

    data_user = click_all.groupby(['item_id'])['user_id', 'time'].agg({
        'user_id': lambda x: ','.join(list(x)), 'time': lambda x: ','.join(list(x))
    }).reset_index()

    for i, row in tqdm(data_user.iterrows()):
        list_user_id = row['user_id'].split(',')
        list_time = row['time'].split(',')
        len_list_user = len(list_user_id)
        for i, (user_i, time_i) in enumerate(zip(list_user_id, list_time)):
            for j, (user_j, time_j) in enumerate(zip(list_user_id, list_time)):
                t = np.abs(float(time_i) - float(time_j))
                d = np.abs(i - j)
                if i < j:
                    if user_i not in user_matrix_association_rules:
                        user_matrix_association_rules[user_i] = {}
                    if user_j not in user_matrix_association_rules[user_i]:
                        user_matrix_association_rules[user_i][user_j] = 0

                    user_matrix_association_rules[user_i][user_j] += 1 * 0.7 * (0.8 ** (d - 1)) * (
                            1 - t * 10000) / np.log(
                        1 + len_list_user)

                if i > j:
                    if user_i not in user_matrix_association_rules:
                        user_matrix_association_rules[user_i] = {}
                    if user_j not in user_matrix_association_rules[user_i]:
                        user_matrix_association_rules[user_i][user_j] = 0

                    user_matrix_association_rules[user_i][user_j] += 1 * 1.0 * (0.8 ** (d - 1)) * (
                            1 - t * 10000) / np.log(
                        1 + len_list_user)

    assert len(user_matrix_association_rules.keys()) == len(set(click_all['user_id']))

    list_item_similar = []
    list_score_similar = []
    list_user_id = []

    print('------- association rules 召回 ---------')
    for i, row in tqdm(data_item.iterrows()):
        list_item_id = row['item_id'].split(',')
        dict_item_id_score = {}
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, score_similar in sorted(item_matrix_association_rules[item_i].items(), key=lambda x: x[1],
                                                reverse=True)[0:k]:
                if item_j not in list_item_id:
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0

                    dict_item_id_score[item_j] += score_similar * (0.7 ** i)

        user_i = row['user_id']
        dict_user_item_id_score = {}
        for user_j, score_similar in sorted(user_matrix_association_rules[user_i].items(), key=lambda x: x[1],
                                            reverse=True)[0:k]:
            row = data_item[data_item['user_id']==user_j]
            list_user_item_id = row['item_id'].values[0].split(',')
            for i, item_i in enumerate(list_user_item_id[::-1]):
                for item_j, score_similar in sorted(item_matrix_association_rules[item_i].items(), key=lambda x: x[1],
                                                    reverse=True)[0:k]:
                    if item_j not in list_user_item_id:
                        if item_j not in dict_user_item_id_score:
                            dict_user_item_id_score[item_j] = 0
                    # print(user_i)
                    # print(user_j)
                    # print(item_j)

                        dict_user_item_id_score[item_j] += score_similar * (0.7 ** i)* user_matrix_association_rules[user_i][user_j]

        dict_total_score = {}
        for key in list(set(list(dict_item_id_score.keys()) + list(dict_user_item_id_score.keys()))):
            # dict_total_score[key] = alpha* dict_item_id_score[key] + (1-alpha) * dict_user_item_id_score[key]
            if key in dict_item_id_score and key in dict_user_item_id_score:
                dict_total_score[key] = alpha * dict_item_id_score[key] + (1 - alpha) * dict_user_item_id_score[key]
            elif key in dict_item_id_score:
                dict_total_score[key] = dict_item_id_score[key]
            elif key in dict_user_item_id_score:
                dict_total_score[key] = dict_user_item_id_score[key]

        for key in list(set(dict_total_score.keys())):
            dict_total_score[key] *= 1.0 / math.log2(2 + stat_cnt[key])

        dict_item_id_score_topk = sorted(dict_total_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 100
                    dict_item_id_score_topk.append((item_similar, score_similar))
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])

        # dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        # dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])

    topk_recall = pd.DataFrame(
        {'user_id': list_user_id, 'item_similar': list_item_similar, 'score_similar': list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall

def metrics_recall(topk_recall, phase, k, sep=10):
    data_ = topk_recall[topk_recall['pred'] == 'train'].sort_values(['user_id', 'score_similar'], ascending=False)
    data_ = data_.groupby(['user_id']).agg(
        {'item_similar': lambda x: list(x), 'next_item_id': lambda x: ''.join(set(x))})

    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in
                      zip(data_['next_item_id'], data_['item_similar'])]

    print('-------- 召回效果 -------------')
    print('--------:phase: ', phase, ' -------------')
    data_num = len(data_)
    for topk in range(0, k + 1, sep):
        hit_num = len(data_[(data_['index'] != -1) & (data_['index'] <= topk)])
        hit_rate = hit_num * 1.0 / data_num
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num : ', hit_num, 'hit_rate : ', hit_rate, ' data_num : ',
              data_num)
        print()

    hit_rate = len(data_[data_['index'] != -1]) * 1.0 / data_num
    return hit_rate

for phase in range(0, now_phase + 1):
    print('phase:', phase)
    click_train = pd.read_csv(
        train_path + '/underexpose_train_click-{phase}.csv'.format(phase=phase)
        , header=None
        , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )
    click_test = pd.read_csv(
        test_path + '/underexpose_test_click-{phase}/underexpose_test_click-{phase}.csv'.format(phase=phase)
        , header=None
        , nrows=nrows
        , names=['user_id', 'item_id', 'time']
        , sep=','
        , dtype={'user_id': np.str, 'item_id': np.str, 'time': np.str}
    )

    click = click_train.append(click_test)

    if flag_append:
        click_all = click_all.append(click)
    else:
        click_all = click

    click_all = click_all.sort_values('time')
    click_all = click_all.drop_duplicates(['user_id', 'item_id', 'time'], keep='last')

    set_pred = set(click_test['user_id'])
    set_train = set(click_all['user_id']) - set_pred

    temp_ = click_all
    temp_['pred'] = temp_['user_id'].map(lambda x: 'test' if x in set_pred else 'train')
    temp_ = temp_[temp_['pred'] == 'train'].drop_duplicates(['user_id'], keep='last')
    temp_['remove'] = 'remove'

    train_test = click_all
    train_test = train_test.merge(temp_, on=['user_id', 'item_id', 'time', 'pred'], how='left')
    train_test = train_test[train_test['remove'] != 'remove']

    print("-------- train_test  -------------")
    # print(train_test.shape)
    # print(train_test.columns)

    dict_label_user_item = dict(zip(temp_['user_id'], temp_['item_id']))

    temp_ = train_test.groupby(['item_id'])['user_id'].count().reset_index()
    temp_ = temp_.sort_values(['item_id'])
    hot_list = list(temp_['item_id'])[::-1]

    print(temp_.shape)
    print('-------- 召回 -------------')
    topK_recall=HybridCF(click_all=train_test, dict_label=dict_label_user_item, k=recall_num)

    print('-------- 评测召回效果 -------------')
    hit_rate = metrics_recall(topk_recall=topk_recall, phase=phase, k=recall_num, sep=int(recall_num / 10))
    print('-------- 召回TOP:{k}时, 命中百分比:{hit_rate} -------------'.format(k=recall_num, hit_rate=hit_rate))

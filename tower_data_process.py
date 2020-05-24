import pandas as pd
import numpy as np
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
train_path = '/Users/xuzhiyuan/Downloads/underexpose_train'
test_path = '/Users/xuzhiyuan/Downloads/underexpose_test'
item_vec = pd.read_csv(train_path + '/underexpose_item_feat.csv',
                       names=['item_id'] + ['txt_vec_{}'.format(i) for i in range(128)] + ['img_vec_{}'.format(i) for i
                                                                                           in range(128)])


## interfere
def interfere():
    import sys
    sys.exit(0)


def save(item, name):
    np.save('{}.npy'.format(name), item)


## item preprocessing
for vec in ['txt_vec', 'img_vec']:
    for i in range(128):
        col = vec + "_" + str(i)
        if i == 0:
            item_vec[col] = item_vec[col].map(lambda x: float(x[1:]))
        elif i == 127:
            item_vec[col] = item_vec[col].map(lambda x: float(x[:-1]))
        else:
            item_vec[col] = item_vec[col].map(lambda x: float(x))

item_vec = item_vec.set_index('item_id')
item_list = item_vec.index.values.tolist()
item_num = max(item_list)
item_vecs = []
for i in range(item_num + 1):
    if i in item_list:
        item_vecs.append(item_vec.loc[i].values.reshape([1, -1]))
    else:
        item_vecs.append(np.array([0] * 256).reshape(1, -1))
item_vecs = np.stack(item_vecs, axis=0).squeeze(axis=1)
save(item_vecs, 'item')
print(item_vecs.shape)
interfere()

now_phase = 6
whole_click = pd.DataFrame()


### 多塔模型

class Sampler:
    def __init__(self, data, item_ids):
        ### user_id, item_id, timestamp
        self.data = data
        self.hot_list = data['item_id'].value_counts()
        self.data_group_sort_ = self.data.sort_values('time').groupby('user_id')
        self.item_list_ = self.data_group_sort_['item_id'].apply(list)
        self.item_time_ = self.data_group_sort_['time'].apply(list)
        self.item_ids = item_ids

    def getNeg(self, user_id, num, cands=500):
        hot = self.hot_list.index[:cands].values.tolist()
        item_list = self.item_list_.loc[user_id]
        cand_list = [c for c in hot if c not in item_list and c in self.item_ids]
        value = data['item_id'].value_counts()
        cand_value = value.loc[cand_list].values.tolist()
        total = sum(cand_value)
        cand_value = [value * 1.0 / total for value in cand_value]
        return random.choices(cand_list, weights=cand_value, k=min(num, len(cand_value)))

    def getPos(self, user_id):
        item_id = self.item_list_.loc[user_id]
        item_time = self.item_time_.loc[user_id]
        l = [[item_id[i], item_time[i], i] for i in range(len(item_id))]
        return [item for item in l if item[0] in self.item_ids]


for c in range(now_phase + 1):
    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c),
                             header=None,
                             names=['user_id', 'item_id', 'time'])

    all_click = click_train.append(click_test)
    whole_click = whole_click.append(all_click)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    ###
    data = whole_click.copy()
    data = data.fillna(0)
    data['user_id'] = data['user_id'].map(lambda x: int(x))
    print(data.columns)

    ###
    samples = []
    sampler = Sampler(data[['user_id', 'item_id', 'time']], item_list)
    for user_id in tqdm(click_test['user_id'].unique()):
        pos_list = sampler.getPos(user_id)
        pos_pair = [pos_list[i] + pos_list[j] for i in range(len(pos_list)) for j in range(i + 1, len(pos_list))]
        neg_list = sampler.getNeg(user_id, num=3 * len(pos_list))
        for pos in pos_pair:
            negs = random.choices(neg_list, k=3)
            pair = pos + [negs[0], negs[1], negs[2]]
            samples.append([user_id] + pair)
    df = pd.DataFrame(samples,
                      columns=['user_id', 'pos1_id', 'pos1_time', 'pos1_loc', 'pos2_id', 'pos2_time', 'pos2_loc',
                               'neg1', 'neg2', 'neg3'])
    df.to_csv('inputs_{}'.format(c), index=False, sep='\t')

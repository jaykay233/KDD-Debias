import pandas as pd
import numpy as np
import faiss
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch import nn as nn
import warnings
warnings.filterwarnings('ignore')
item_vec = np.load('item.npy').astype('float32')
d = 256
index = faiss.IndexFlatL2(d)
index.add(item_vec)
from tower import TripleDSSM
model = TripleDSSM.load_from_checkpoint('epoch=0.ckpt',)
model.eval()

print(index.is_trained)
print(index.ntotal)


warnings.filterwarnings('ignore')
train_path = '/Users/xuzhiyuan/Downloads/underexpose_train'
test_path = '/Users/xuzhiyuan/Downloads/underexpose_test'


now_phase = 6
whole_click = pd.DataFrame()
res = []
for c in range(now_phase + 1):
    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
                              names=['user_id', 'item_id', 'time'])
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_click-{}.csv'.format(c, c),
                             header=None,
                             names=['user_id', 'item_id', 'time'])
    all_click = click_train.append(click_test)
    whole_click = whole_click.append(all_click)
    whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
    data = whole_click.sort_values('time').groupby('user_id').agg(list)
    click_qtime = pd.read_csv(test_path + '/underexpose_test_click-{}/underexpose_test_qtime-{}.csv'.format(c,c),header=None,
                              names=['user_id','time'])
    user_ids = []
    times = []
    locations = []
    for _, row in click_qtime.iterrows():
        user_id = int(row['user_id'])
        time = row['time']
        item_list = data.loc[user_id]
        loc = len(item_list)

        user_id = torch.tensor(user_id).long()
        time = torch.tensor(time).float()
        loc = torch.tensor(loc).long()
        user_ids.append(user_id)
        times.append(time)
        locations.append(loc)
    user_ids = torch.from_numpy(np.array(user_ids).reshape(-1)).long()
    times = torch.from_numpy(np.array(times).reshape(-1,1)).float()
    locations = torch.from_numpy(np.array(locations).reshape(-1)).long()

    with torch.no_grad():
        embeddings = model.get_user_embedding(user_id=user_ids,time=times,location=locations).numpy()
        _,I = index.search(embeddings,100)
        res.extend(I)







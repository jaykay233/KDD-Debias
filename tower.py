import os

import torch
from torch import nn as nn
from torch.autograd import Variable
import faiss
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

item_vecs = np.load('item.npy')


class Clickset(Dataset):
    def __init__(self, phase_start, phase_end):
        self.paths = ['inputs_{}.csv'.format(i) for i in range(phase_start, phase_end + 1)]
        self.data = []
        for path in self.paths:
            data = pd.read_csv(path, sep='\t')
            self.data.append(data)
        self.data = pd.concat(self.data, axis=0).sample()

    def __getitem__(self, index):
        data = self.data[index]
        user_id = data[0]
        pos1_id = data[1]
        pos1_time = data[2]
        pos1_location = data[3]
        pos2_id = data[4]
        pos2_time = data[5]
        pos2_location = data[6]
        neg1 = data[7]
        neg2 = data[8]
        neg3 = data[9]
        return user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3

    def __len__(self):
        return len(self.data)


TrainSet = Clickset(0, 5)
TestSet = Clickset(6, 6)


class TripleDSSM(pl.LightningModule):
    """
    inputs: user_id, pos_a , pos_b, neg[list](neg[1],neg[2],neg[3])
    pos_a: (id, time, location)
    pos_b: (id, time, location)
    """

    def __init__(self):
        super(TripleDSSM, self).__init__()
        self.item_embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(item_vecs), freeze=True)

        self.loc_embedding = torch.nn.Embedding(1000, 256)
        self.time_embedding = torch.nn.Linear(1, 256)
        self.user_embedding = torch.nn.Embedding(200000, 256)
        self.linear = nn.Sequential(torch.nn.Linear(256, 256), nn.Tanh(), torch.nn.Linear(256, 256))
        self.dis = torch.cosine_similarity

    def forward(self, user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3):
        user_embs = self.user_embedding(user_id)
        # print(pos1_id)
        # print(pos2_id)
        # print(item_vecs.shape)
        pos1_embs = self.item_embedding(pos1_id)
        pos2_embs = self.item_embedding(pos2_id)
        neg1_embs = self.item_embedding(neg1)
        neg2_embs = self.item_embedding(neg2)
        neg3_embs = self.item_embedding(neg3)

        pos1_time = pos1_time.reshape(-1,1).float()
        pos2_time = pos2_time.reshape(-1,1).float()
        # print(pos1_time.shape)

        time1_embs = self.time_embedding(pos1_time)
        time2_embs = self.time_embedding(pos2_time)

        loc1_embs = self.loc_embedding(pos1_location)
        loc2_embs = self.loc_embedding(pos2_location)
        user_fusioned1 = self.linear(user_embs + time1_embs + loc1_embs)
        user_fusioned2 = self.linear(user_embs + time2_embs + loc2_embs)
        dis_pos1 = self.dis(user_fusioned1, pos1_embs)
        dis_neg_avg1 = (self.dis(user_fusioned1, neg1_embs) + self.dis(user_fusioned1, neg2_embs) + self.dis(
            user_fusioned1, neg3_embs)) / 3
        dis_pos2 = self.dis(user_fusioned2, pos2_embs)
        dis_neg_avg2 = (self.dis(user_fusioned2, neg1_embs) + self.dis(user_fusioned2, neg2_embs) + self.dis(
            user_fusioned2, neg3_embs)) / 3
        # print(dis_pos1)
        # print(dis_neg_avg1)
        # print(dis_pos2)
        # print(dis_neg_avg2)
        loss =  torch.max(-dis_pos1 + dis_neg_avg1 + 0.2, 0) + torch.max(-dis_pos2 + dis_neg_avg2 + 0.2, 0)
        return sum(loss)


    def get_user_embedding(self, user_id, time, location):
        user_embs = self.user_embedding(user_id)
        time_embs = self.time_embedding(time)
        location_embs = self.loc_embedding(location)
        return self.linear(user_embs + time_embs + location_embs)

    def train_dataloader(self):
        return DataLoader(TrainSet, batch_size=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TestSet, batch_size=4)

    def training_step(self, batch, batch_idx):
        user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3 = batch
        user_id = user_id.long()
        pos1_id = pos1_id.long()
        pos1_location = pos1_location.long()
        pos2_id = pos2_id.long()
        pos2_location = pos2_location.long()
        neg1 = neg1.long()
        neg2 = neg2.long()
        neg3 = neg3.long()

        loss = self(user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3 = batch
        user_id = user_id.long()
        pos1_id = pos1_id.long()
        pos1_location = pos1_location.long()
        pos2_id = pos2_id.long()
        pos2_location = pos2_location.long()
        neg1 = neg1.long()
        neg2 = neg2.long()
        neg3 = neg3.long()

        loss = self(user_id, pos1_id, pos1_time, pos1_location, pos2_id, pos2_time, pos2_location, neg1, neg2, neg3)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)


from pytorch_lightning import Trainer
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

model = TripleDSSM()
trainer = Trainer(max_epochs=5,checkpoint_callback=checkpoint_callback,early_stop_callback=early_stop_callback,gpus=0)
# trainer.fit(model)

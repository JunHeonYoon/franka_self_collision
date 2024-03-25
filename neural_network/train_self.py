from __future__ import division
import os
import time
import torch
import argparse
import pickle
import numpy as np
from models_self import SelfCollNet
import datetime as dt
import shutil

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm

class CollisionNetDataset(Dataset):
    """
    data pickle contains dict
        'q'          : joint angle
        'normalize_q': normalized joint angle
        'nerf_q'     : nerf joint angle
        'coll'       : collision vector data (coll: 1, free: 0)
        'min_dist'   : minimum distance btw robot links (coll: -1)
    """
    def __init__(self, file_name,):
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
            self.q = dataset['q']
            self.coll = dataset['coll']
            self.min_dist = dataset['min_dist']*100 # meter to centi-meter
        print('q shape: ', self.q.shape)
        print('min_dist shape: ', self.min_dist.shape)

    def __len__(self):
        return len(self.min_dist)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
    
        return np.array(self.q[idx],dtype=np.float32), np.array(self.min_dist[idx],dtype=np.float32)

def main(args):
    train_ratio = 0.95
    test_ratio = 0.002
    
    date = dt.datetime.now()
    data_dir = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}/".format(date.year, date.month, date.day, date.hour, date.minute,date.second)
    log_dir = 'log/self/' + data_dir
    chkpt_dir = 'model/checkpoints/self/' + data_dir
    model_dir = 'model/self/' + data_dir

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    folder_path = 'model/checkpoints/self/'
    num_save = 3
    order_list = sorted(os.listdir(folder_path), reverse=True)
    remove_folder_list = order_list[num_save:]
    for rm_folder in remove_folder_list:
        shutil.rmtree('log/self/'+rm_folder)
        shutil.rmtree('model/checkpoints/self/'+rm_folder)
        shutil.rmtree('model/self/'+rm_folder)
    
    suffix = 'rnd{}'.format(args.seed)

    file_name = "dataset/2024_03_25_14_29_10/dataset.pickle"

    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}'.format(suffix)

    """
    layer size = [7, (21 if nerf),  hidden1, hidden2, , ..., 1(mininum dist)]
    """
    layer_size = [7, 128, 64, 1]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    print('loading data ...')
    read_time = time.time()
    dataset = CollisionNetDataset(file_name=file_name)
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    val_size = len(dataset) - (train_size + test_size)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    

    def loss_fn_fc(y_hat, y):
        RMSE = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        return RMSE
    

    collnet = SelfCollNet(fc_layer_sizes=layer_size,
                          batch_size=args.batch_size,
                          device=device,
                          nerf=True).to(device)
    print(collnet)

    optimizer = torch.optim.Adam(collnet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                           threshold=0.01, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-04, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # clear log
    with open(log_file_name, 'w'):
        pass

    min_loss = 1e100
    e_notsaved = 0

    for q, min_dist in test_data_loader:
        test_q, test_min_dist = q.to(device).squeeze(), min_dist.to(device).squeeze()

    for epoch in range(args.epochs):
        loader_tqdm = tqdm.tqdm(train_data_loader)

        for q, min_dist in loader_tqdm:
            train_q, train_min_dist = q.to(device).squeeze(),  min_dist.to(device).squeeze()
            
            collnet.train()
            with torch.cuda.amp.autocast():
                min_dist_hat= collnet.forward(train_q)
                min_dist_hat = min_dist_hat.squeeze()
                loss_fc_train = loss_fn_fc(min_dist_hat, train_min_dist)
                loss_train = loss_fc_train

            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


            
        collnet.eval()
        with torch.cuda.amp.autocast():
            min_dist_hat = collnet.forward(test_q)
            min_dist_hat = min_dist_hat.squeeze()
            loss_fc_test = loss_fn_fc(min_dist_hat, test_min_dist)
            loss_test = loss_fc_test
        scheduler.step(loss_fc_test)
        
        coll_hat_bin = min_dist_hat < 0.05
        coll_bin = test_min_dist < 0.05

        test_accuracy = (coll_bin == coll_hat_bin).sum().item() / test_min_dist.size(dim=0)

        truth_positives = (coll_bin == 1).sum().item() 
        truth_negatives = (coll_bin == 0).sum().item() 

        confusion_vector = (coll_hat_bin / coll_bin)
        
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        accuracy = ({'tp': true_positives / truth_positives,
                     'fp': false_positives / truth_negatives,
                     'tn': true_negatives / truth_negatives,
                     'fn': false_negatives / truth_positives})

        if epoch == 0:
            min_loss = loss_test

        scheduler.step(loss_test)

        if loss_test < min_loss:
            e_notsaved = 0
            print('saving model', loss_test.item())
            checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{}_self'.format(loss_test.item(), model_name, epoch, args.seed) + '.pkl'
            torch.save(collnet.state_dict(), checkpoint_model_name)
            min_loss = loss_test
        print("Epoch: {} (Saved at {})".format(epoch, epoch-e_notsaved))
        print("[Train] fc loss  : {:.3f}".format(loss_train.item()))
        print("[Test]  fc loss  : {:.3f}".format(loss_test.item()))
        print("[Test]  Accuracy : {}".format(test_accuracy))
        print("min_dist         : {}".format(test_min_dist.detach().cpu().numpy()[:2]))
        print("min_dist_hat     : {}".format(min_dist_hat.detach().cpu().numpy()[:2]))
        print("=========================================================================================")

        with open(log_file_name, 'a') as f:
            f.write("Epoch: {} (Saved at {}) / Train Loss: {} / Test Loss: {} / Test Accuracy: {} / TP: {} / FP: {} / TN: {} / FN: {}".format(epoch,
                                                                                                                                              epoch - e_notsaved,
                                                                                                                                              loss_train,
                                                                                                                                              loss_test,
                                                                                                                                              test_accuracy,
                                                                                                                                              [accuracy["tp"]],
                                                                                                                                              [accuracy["fp"]],
                                                                                                                                              [accuracy["tn"]],
                                                                                                                                              [accuracy["fn"]]))

        e_notsaved += 1
    torch.save(collnet.state_dict(), model_dir+'ss{}.pkl'.format(model_name))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    
    args = parser.parse_args()
    main(args)
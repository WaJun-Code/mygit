# coding: UTF-8
import argparse,random,os,tqdm,gc

import torch
from torch.utils.data import DataLoader
import numpy as np

from data_loader import DADataSet
from model import BaselineModel
from sklearn.metrics import roc_auc_score, f1_score   #f1_score(labels,y)，roc_auc_score(labels,logits)

def seed_torch(seed=1029):   #torch使结果可复现
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed=2022  #0 2 4
seed_torch(seed)
def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--data_path', default='../train.json', type=str)
    parser.add_argument('--num_epochs', default=12, type=int, help='the epoch of train')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size of dataset')
    parser.add_argument('--lr', default=3e-5, type=float, help='the learning rate of bert')
    parser.add_argument('--bert_path', default='./chinese-bert-wwm-ext')  #ernie_gram
    parser.add_argument('--warm_up', default=0.01)
    device = 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    return parser.parse_args()
def find_best_threshold(y_valid, oof_prob):
    best_f1 = 0
    
    for th in [i/500 for i in range(5, 475)]:  #tqdm()
        oof_prob_copy = oof_prob.copy()
        oof_prob_copy[oof_prob_copy >= th] = 1
        oof_prob_copy[oof_prob_copy < th] = 0

        f1=0
        for i in range(oof_prob.shape[1]):
            f1 += f1_score(y_valid[:,i], oof_prob_copy[:,i])
        if f1 > best_f1:
            best_th = th
            best_f1 = f1
  
        gc.collect()
        
    return best_th, best_f1/oof_prob.shape[1]

def valid(model,valid_iter):
    # with open(args.data_path, 'r', encoding='utf8') as f:
    #     lines = f.readlines()
    logits,T = [],[]
    total_iter = 0
    valid_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (text, mask, labels) in enumerate(tqdm.tqdm(valid_iter)):
            outputs = model(text, mask)
            loss_function = torch.nn.BCELoss()
            loss = loss_function(outputs, labels)
            logits += outputs.detach().cpu().numpy().tolist()
            T += labels.detach().cpu().numpy().tolist()
            total_iter += 1
            valid_loss += loss.data
    valid_loss = valid_loss / total_iter
    print(f'[valid_Loss]: {valid_loss}')
    logits,T = np.array(logits),np.array(T)
    auc = roc_auc_score(T,logits)
    print(f'[valid_auc]: {auc}')
    best_t,f1 = find_best_threshold(T, logits)
    print(f'valid_best_th:{best_t},[valid_f1]: {f1}')
    return best_t,f1

args = get_args()
full_dataset = DADataSet(args)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
valid_iter = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True )
model = BaselineModel('./chinese-bert-wwm-ext')
model.load_state_dict(torch.load("baseline_model.pt", map_location='cpu'))
model.eval()
valid(model,valid_iter)
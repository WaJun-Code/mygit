# coding: UTF-8
import argparse
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
from tqdm import tqdm
import transformers
from torch.utils.data import DataLoader
import numpy as np
import os,random,gc

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
seed=42  #0 2 4
seed_torch(seed)
def get_args():
    parser = argparse.ArgumentParser(description='The model for disease diagnosis.')
    parser.add_argument('--data_path', default='../train.json', type=str)
    parser.add_argument('--num_epochs', default=6, type=int, help='the epoch of train')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size of dataset')
    parser.add_argument('--lr', default=3e-5, type=float, help='the learning rate of bert')
    parser.add_argument('--bert_path', default='./chinese-bert-wwm-ext')  #ernie_gram
    parser.add_argument('--warm_up', default=0.01)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    return parser.parse_args()
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)
        return loss
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def adjust_lr(step):
    three_epoch_step = 5*num_training_steps/args.num_epochs
    if step < three_epoch_step:
        gamma = ((5E-6/args.lr) - 1) * (step/three_epoch_step) + 1
    else:
        gamma = 5E-6/args.lr
    return gamma
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
    logits,T = [],[]
    total_iter = 0
    valid_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (text, mask, labels) in enumerate(valid_iter):
            outputs = model(text, mask)
            loss_function = nn.BCELoss()
            loss = loss_function(outputs, labels)
            logits += outputs.detach().cpu().numpy().tolist()
            T += labels.detach().cpu().numpy().tolist()
            total_iter += 1
            valid_loss += loss.data
            torch.cuda.empty_cache()
    valid_loss = valid_loss / total_iter
    print(f'[valid_Loss]: {valid_loss}')
    logits,T = np.array(logits),np.array(T)
    auc = roc_auc_score(T,logits)
    print(f'[valid_auc]: {auc}')
    best_t,f1 = find_best_threshold(T, logits)
    print(f'valid_best_th:{best_t},[valid_f1]: {f1}')
    return best_t,f1

if __name__ == '__main__':
    args = get_args()
    full_dataset = DADataSet(args)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True )          #full
    valid_iter = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True )

    model = BaselineModel(args.bert_path).to(args.device)
    #model = nn.DataParallel(model)
    num_training_steps = len(train_iter) * args.num_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = RAdam(optimizer_grouped_parameters, lr=args.lr, eps=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = adjust_lr)

    # optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    # lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up * num_training_steps, num_training_steps=num_training_steps)

    # ema = EMA(model, 0.999)
    # ema.register()

    loss_function = nn.BCELoss()
    best_f1 = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_iter = 0
        total_loss = 0.
        for i, (text, mask, labels) in enumerate(tqdm(train_iter)):
            model.zero_grad()
            outputs,loss0 = model(text, mask,True)
            loss = loss_function(outputs, labels)+loss0
            loss.backward()
            optimizer.step()
            # ema.update()
            lr_scheduler.step()
            total_iter += 1
            total_loss += loss.data
            torch.cuda.empty_cache()
        avg_loss = total_loss / total_iter
        print(f'{epoch}[train_Loss]: {avg_loss}')
        # ema.apply_shadow()
        th,f1 = valid(model,valid_iter)
        if best_f1<f1:#-0.01 or (best_f1<f1 and th<0.6)
            best_f1 = f1
            # torch.save(model.state_dict(), 'baseline_model.pt')
            print(f'save_best_th:{th},[save_f1]: {f1}')
            # print('Save Best Model...')


#import jieba
#from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer, CountVectorizer
# 做tf-idf提取重要词汇的操作：
# data = pd.read_csv("data.tsv",sep='\t',header=0,names=["q1","q2","label"]).dropna(axis=0,how='any')
# length = data['q1'].apply(len).sort_values().tolist()
# print(length[-2000:])   #此时对应45长度

# doc_list = data.iloc[:,0].values.tolist()
# corpus=[]
# for i in tqdm.tqdm(doc_list):
#     s=''
#     for j in list(jieba.cut(i)):s = s+j+' '   #分词
#     corpus.append(s)
# print(corpus[:3])
# print(doc_list[246406],doc_list[246407])
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(corpus[:3])
# print(tfidf_matrix.toarray())
# print(tfidf_vectorizer.get_feature_names())   #通过该矩阵+list可确定一个句子里的重要词汇是哪些

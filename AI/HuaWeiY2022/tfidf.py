# coding: UTF-8
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from tqdm import tqdm

import numpy as np
import os,random,gc,json,re,jieba


from sklearn.metrics import roc_auc_score, f1_score   #f1_score(labels,y)，roc_auc_score(labels,logits)
import numpy as np
import pandas as pd

with open("labels0.txt", 'r', encoding='utf8') as f:s0 = f.readlines()
s0 = [x.strip() for x in s0]

class TFIDF(object):
    """TFIDF简单实现"""
    def __init__(self, corpus):
        self.word_id = {}   #每个词被依次遍历的顺序
        self.corpus = corpus
        self.smooth_idf = 0.01

    def fit_transform(self, corpus):
        pass    
        
    def get_vocabulary_frequency(self):
        # 统计各词出现个数
        id = 0
        for single_corpus in self.corpus:
            for word in single_corpus:
                # 如果该词不在词汇统计表的key中
                if word not in self.word_id:
                    # 将该词放入到词汇统计表及词汇编码表中
                    self.word_id[word] = id
                    id += 1

        # 生成corpus长度 X 词频统计表长度的全零矩阵
        X = np.zeros((len(self.corpus), len(self.word_id)))
        for i in range(X.shape[0]):
            single_corpus = self.corpus[i]
            L = len(single_corpus)
            for word in single_corpus:
                X[i,self.word_id[word]] += 1/L

        return X
    def get_tf_tdf(self):
        """
        计算idf并生成最后的TFIDF矩阵
        """       
        X = self.get_vocabulary_frequency()
        num_samples, n_features = X.shape
        df = []
        for i in range(n_features):
            # 统计每个特征的非0的数量，也就是逆文档频率指数的分式中的分母，是为了计算idf
            # bincount: 传入一个数组, 返回对应索引出现的次数的数组(大致可以这么理解)
            df.append(num_samples - (X[:,i]==0).sum() )
        df = np.array(df)
        np.savetxt("df.txt",df,fmt="%d")
        # 是否需要添加平滑因子
        df += int(self.smooth_idf)
        num_samples += int(self.smooth_idf)
        idf = np.log(num_samples / df) + 1 # 核心公式
        return X*idf
class new_TFIDF(object):
    """TFIDF简单实现"""
    def __init__(self, corpus,T):
        self.word_id = {}   #每个词被依次遍历的顺序
        self.XT = pd.concat((pd.DataFrame(["".join(i) for i in corpus],columns=["corpus"]),pd.DataFrame(T,columns=[f"T_{i}" for i in range(len(T[0]))]) ),axis=1)
        dfx = []
        for T_x in [f"T_{i}" for i in range(len(T[0]))]:
            dfx.append("".join(self.XT[self.XT[T_x]==1]["corpus"].values.tolist()) )
        self.XT = dfx  #下标为对应label，内容为相应词条
        self.smooth_idf = 0.01

    def fit_transform(self, corpus):
        pass
        
    def get_vocabulary_frequency(self):
        # 统计各词出现个数
        id = 0
        for single_corpus in self.XT:
            for word in single_corpus:
                # 如果该词不在词汇统计表的key中
                if word not in self.word_id:
                    # 将该词放入到词汇统计表及词汇编码表中
                    self.word_id[word] = id
                    id += 1

        # 生成corpus长度 X 词频统计表长度的全零矩阵
        X = np.zeros((len(self.XT), len(self.word_id)))
        for i in range(X.shape[0]):
            single_corpus = self.XT[i]
            L = len(single_corpus)
            for word in single_corpus:
                X[i,self.word_id[word]] += 1/L

        return X
    def get_tf_tdf(self):
        """
        计算idf并生成最后的TFIDF矩阵
        """       
        X = self.get_vocabulary_frequency()
        num_samples, n_features = X.shape
        df = []
        for i in range(n_features):
            # 统计每个特征的非0的数量，也就是逆文档频率指数的分式中的分母，是为了计算idf
            # bincount: 传入一个数组, 返回对应索引出现的次数的数组(大致可以这么理解)
            df.append(num_samples - (X[:,i]==0).sum() )
        df = np.array(df)
        # 是否需要添加平滑因子
        df += int(self.smooth_idf)
        num_samples += int(self.smooth_idf)
        idf = np.log(num_samples / df) + 1 # 核心公式
        #再引入11089个词条对应的idf
        idf1 = json.load(open("idf1.json","r"))
        idf1 = np.array([idf1[word] for word in self.word_id.keys()])
        idf1 = np.log(11089 / idf1) + 1
        return X*idf*idf1
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
def supplementary_cut(x): #先删掉“否认”"无"部分，再直接匹配
    L = set()
    x = x.strip().replace("，","。").split("。")
    x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
    n = len(x)
    for i in range(n):
        if "示" in x[i]:
            L.add(x[i])
    return "。".join(L)
def past_cut(x): #先删掉“否认”"无"部分，再直接匹配
    L = set()
    x = x.strip().replace("，","。").split("。")
    #删掉含“血压”的句子
    i = 0
    while i<len(x):
        if "否认" in x[i] or "无" in x[i]:
            x.pop(i)
            continue
        i+=1
    x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
    n = len(x)
    for i in range(n):
        if "有" in x[i] or "诊断" in x[i]:
            L.add(x[i])
    return "。".join(L)
def present_cut(x):   #df["history_of_present_illness"].apply(str_cut)
    L = set()
    x = x.strip().replace("，","。").split("。")
    #删掉含“血压”的句子
    i = 0
    while i<len(x):
        if "血压" in x[i]:
            x.pop(i)
            continue
        i+=1
    x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
    n = len(x)
    for i in range(n):
        if "收" in x[i]:
            if i>0 and "收" not in x[i-1]:L.add(x[i-1])
            L.add(x[i])
            if i<n-1 and "收" not in x[i+1]:L.add(x[i+1])
        elif "诊" in x[i]:
            L.add(x[i])
            if i<n-1 and "诊" not in x[i+1]:L.add(x[i+1])
    return "。".join(L)
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

def valid(model,valid_iter,T_tfidf):
    logits,T = [],[]
    total_iter = 0
    valid_loss = 0.
    
    with torch.no_grad():
        # 建立计算图
        total_iter = 0
        total_loss = 0.
        for i, (X, labels) in enumerate(valid_iter):
            outputs = model(X,T_tfidf)
            loss_function = nn.BCELoss()
            loss = loss_function(outputs, labels[:,:52]-labels[:,52:])
            logits += (outputs+labels[:,52:]).detach().cpu().numpy().tolist()
            T += labels[:,:52].detach().cpu().numpy().tolist()
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
class Data(Dataset):
    def __init__(self, X,T):
        self.X = X
        self.T = T
    def __getitem__(self, idx):
        return self.X[idx,:].to(device), self.T[idx,:].to(device)
    def __len__(self):
        return len(self.T)
class LR(nn.Module):
    def __init__(self,input_dim):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, X,Ttfidf):
        X = X.unsqueeze(1).repeat(1,52,1)*Ttfidf.unsqueeze(0).repeat(X.shape[0],1,1)
        out = self.fc(X).squeeze(-1)
        out = torch.sigmoid(out)  #用52个label的embedding来做cosine
        return out
if __name__ == '__main__':
    def clean_x(data):
        data["history_of_present_illness"] = present_cut(data["history_of_present_illness"])
        data["past_history"] = past_cut(data["past_history"])
        if data["supplementary_examination"] == None:data["supplementary_examination"] = "暂缺"   #零填
        data["supplementary_examination"] = supplementary_cut(data["supplementary_examination"])
        data["chief_complaint"] = "。".join(re.findall('[\u4e00-\u9fa5]+',data["chief_complaint"]) )
        data["physical_examination"] = "。".join(re.findall('[\u4e00-\u9fa5]+',data["physical_examination"]) )

        present_illness = data["history_of_present_illness"]+data["past_history"] +data["supplementary_examination"]#+data["chief_complaint"]+data["physical_examination"]
        return re.findall('[\u4e00-\u9fa5]',present_illness)

    with open('labels.txt', 'r', encoding='utf8') as f:labels = f.readlines()
    label2id = {}
    for idx, label in enumerate(labels):
        label2id[label.strip()] = idx
    def _get_label_ids(targets):
        label = [0] * 52
        for target in targets:
            if target in label2id:
                idx = label2id[target]
                label[idx] = 1
        return label

    def post_process(x):
        post_out = torch.zeros([len(s0),])
        for i in range(len(s0)):
            if s0[i] in x:post_out[i] += 0.8
            else:
                for x0 in list(jieba.cut(s0[i])):
                    if x0 in x:post_out[i] += 0.2
        max = post_out.max()
        max = 1 if max==0 else max
        return list(post_out/max)

    with open('../train.json', 'r', encoding='utf8') as f:lines = f.readlines()
    T = [_get_label_ids(json.loads(item)['diagnosis']) for item in lines]
    lines = [clean_x(json.loads(item) ) for item in lines]
    print(max([len(item) for item in lines]))
    print(len(lines) )
    post_out = [post_process(item) for item in lines]

    test = TFIDF(lines)
    X_tfidf,word_id0 = test.get_tf_tdf(),test.word_id
    idf1 = np.loadtxt("df.txt")
    idf1 = dict(zip(word_id0,idf1))  #按【字：idf1】存好了
    json.dump(idf1,open("idf1.json","w"))

    test = new_TFIDF(lines,T)
    T_tfidf,word_id = test.get_tf_tdf().tolist(),test.word_id   #找到对应的最大几个的下标，映射到word_id上，即为label对应的热词
    word_id = list(word_id.keys())
    T_tfidf0 = []
    for T_i in T_tfidf:
        D = dict(zip(word_id,T_i))
        #print(sorted(D.items(),key=lambda x:x[1],reverse=True)[:100])
        T_tfidf0.append([D[word] for word in word_id0] )   #变换为与之前一样的下标
    #得到52个label对应的tf-idf编码embedding后，录入model做点积

    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LR(X_tfidf.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_function = nn.BCELoss()
    T = torch.FloatTensor(T).to(device)
    T_tfidf = torch.FloatTensor(T_tfidf0).to(device)
    post_out = torch.FloatTensor(post_out).to(device)
    T = torch.cat((T,post_out),dim=1)
    X_tfidf = torch.FloatTensor(X_tfidf).to(device)

    full_dataset = Data(X_tfidf, T)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_iter = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True )          #full
    valid_iter = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True )

    epochs = 100
    for epoch in range(epochs):
        # 建立计算图
        total_iter = 0
        total_loss = 0.
        for i, (X, labels) in enumerate(tqdm(train_iter)):
            outputs = model(X,T_tfidf)
            loss = loss_function(outputs, labels[:,:52]-labels[:,52:])
            loss.backward()
            optimizer.step()
            total_iter += 1
            total_loss += loss.data
            torch.cuda.empty_cache()
        print(f'{epoch}[train_Loss]: {total_loss / total_iter}')
        th,f1 = valid(model,valid_iter,T_tfidf)

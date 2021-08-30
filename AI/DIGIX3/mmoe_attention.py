import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm,time,gc
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, auc
import math
ACTION_LIST=['watch_label','is_share','is_watch',  'is_collect', 'is_comment','age', 'city_level']
Dense_fea=['video_score', 'video_duration','video_release_date','video_director_list0', 'video_actor_list0','video_second_class0'] #, 'age0', 'age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'age7', 'gender0', 'gender1', 'gender2', 'gender3', 'country0', 'country1', 'country2',city_level
Sparse_fea={'province':[32,32], 'city':[337,32] , 'device_name':[1394,32],'video_id':[24989,32],'video_name':[23892,32] }
multi_fea={'video_second_class':[146, 9, 32],'video_director_list':[28242+1, 9, 32], 'video_actor_list':[73577+1, 9, 32]}
embed_fea={'video_tags':32}   #, 'video_description':32
# def multi_category_focal_loss1(y_true,y_pred,alpha, gamma=2.0):
#     """model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)"""
#     epsilon = 1.e-7
#     alpha = tf.transpose(tf.expand_dims(tf.constant(alpha, dtype=tf.float32),-1),perm=[1,0])
#     gamma = float(gamma)
#     def multi_category_focal_loss1_fixed(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#         y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
#         ce = -tf.log(y_t)
#         weight = tf.pow(tf.subtract(1., y_t), gamma)
#         print(tf.multiply(weight, ce).shape)
#         print(alpha.shape)
#         fl = tf.matmul(tf.multiply(weight, ce), alpha)
#         loss = tf.reduce_mean(fl)
#         return loss
#     return multi_category_focal_loss1_fixed(y_true,y_pred)
import random,os
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
seed_torch(2021)
class focal_loss(nn.Module):     #alpha越大，结果越偏向于该样本
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==2 
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(2)
            self.alpha[0] += 1-alpha
            self.alpha[1] += alpha
        self.gamma = gamma
    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds_softmax = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # preds_softmax = F.softmax(preds_softmax, dim=1) 
        preds_logsoft = torch.log(preds_softmax+1e-9)
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class MutiHead_attention(nn.Module):
    def __init__(self, model_dim, dq, dk, dv, head_num, drop_out_p):
        super().__init__()

        self.head_num = head_num
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.model_dim = model_dim

        self.WQ = nn.Linear(model_dim, dq*head_num) # head_num个attention模块
        self.WK = nn.Linear(model_dim, dk*head_num)
        self.WV = nn.Linear(model_dim, dv*head_num)
        self.softmax = nn.Softmax(dim = -1)
        self.WO = nn.Linear(dv*head_num, model_dim)
        self.attn_drop = nn.Dropout( p = drop_out_p)
    def forward(self, inputQ, inputK, inputV, mask):
        # input: [batch_size, seq_len, model_dim]
        _Q = self.WQ(inputQ).reshape(inputQ.shape[0], inputQ.shape[1], self.head_num, self.dq) # [batch_size, seq_len, dq*head_num] reshape to [batch_size, seq_len, head_num, dq]
        _K = self.WK(inputK).reshape(inputK.shape[0], inputK.shape[1], self.head_num, self.dk) # [batch_size, seq_len, dk*head_num] reshape to [batch_size, seq_len, head_num, dk]
        _V = self.WV(inputV).reshape(inputV.shape[0], inputV.shape[1], self.head_num, self.dv) # [batch_size, seq_len, dv*head_num] reshape to [batch_size, seq_len, head_num, dv]
        _Q, _K, _V = _Q.permute(0, 2, 1, 3), _K.permute(0, 2, 1, 3), _V.permute(0, 2, 1, 3) # [batch_size, head_num, seq_len, d_{q,k,v}]
        attention_value = self.attention(_Q, _K, _V, mask) # [batch_size, seq_len, head_num, dv]
        attention_value = attention_value.reshape(attention_value.shape[0], attention_value.shape[1], attention_value.shape[2]*attention_value.shape[3]) # 把num_head个dv concat起来
        attention_result = self.WO(attention_value) # [batch_size, seq_len, model_dim]
        attention_result = self.attn_drop(attention_result)
        return attention_result
    def attention(self, Q, K, V, mask = None):
        # Q K V: [batch_size, head_num, seq_len, d_{q,k,v}]
        score = torch.matmul(Q, K.permute(0, 1, 3, 2)) # [batch_size, head_num, seq_len, dq] * [batch_size, head_num, dk, seq_len] = [batch_size, head_num, seq_len, seq_len]
        # 这里可以看做是：取第一组数据，大小是[head_num, seq_len, seq_len]，每一个head对应[seq_len, seq_len]，此矩阵的i行j列即为第i个词对j的注意力分数大小，总共seq_len*seq_len个，多个head即为多头注意力机制
        if mask != None:
            score = score + -1e9*mask
        score = score/math.sqrt(self.dq)
        attention_score = self.softmax(score)
        result = torch.matmul(attention_score, V) # [batch_size, head_num, seq_len, seq_len]*[batch_size, head_num, seq_len, dv] = [batch_size, head_num, seq_len, dv]
        result = result.permute(0,2,1,3) # [batch_size, seq_len, head_num, dv]

        return result
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim = 64, head_num = 4, drop_out_p = 0.4):
        super().__init__()
        self.ff_dim = 128  # model_dim = 32的4倍，论文中的数值d_ff
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.attention_layer = MutiHead_attention(model_dim, hidden_dim, hidden_dim, hidden_dim, head_num, drop_out_p)
        self.Layernorm1 = nn.LayerNorm(model_dim)
        self.FFN =  nn.Sequential(nn.Linear(model_dim, self.ff_dim),
                                  nn.ReLU(),
                                  nn.Linear(self.ff_dim, model_dim))
        self.FFN_dropout =  nn.Dropout( p = drop_out_p)
        self.Layernorm2 = nn.LayerNorm(model_dim)
    def forward(self, input, mask):
        attention_output = self.attention_layer(input, input, input, mask)
        output1 = self.Layernorm1(input + attention_output)
        output2 = self.FFN(output1)
        output2 = self.FFN_dropout(output2)
        result = self.Layernorm2(output2 + output1)
        return result
class MMoE(nn.Module):
    ##### by Kang: 参数增加了Multifea 和 padding_idx
    def __init__(self, Densefea, Sparsefea, Multifea, word2vec_fea, hidden_dim, tasks_dim, device, num_expert = 8, padding_idx = 0): # 默认两个任务，tasks_dim = [10, 2]
        super().__init__()
        self.sparse=list(Sparse_fea.keys())

        ###### by Kang
        self.multi = list(multi_fea.keys())
        self.word2vec_fea = list(embed_fea.keys())
        ######

        self.dense = Densefea
        # self.multi=list(multifea.keys()),list(embedfea.keys()),self.embfea
        self.input_dim =  len(Densefea)
        ###### by Kang
        self.encoder_input_length = len(Sparsefea) + sum([value[1] for value in Multifea.values()]) + len(word2vec_fea)
        ######
        self.Embed = nn.ModuleList([torch.nn.Linear(1, 1, bias=True) for i in range(len(self.sparse))])
        i=0
        for key,value in Sparsefea.items():
            self.input_dim +=value[1]
            self.Embed[i]=nn.Embedding(value[0], value[1])
            i+=1

        ########################### by Kang
        self.Multi_Embed = nn.ModuleList([torch.nn.Linear(1, 1, bias=True) for i in range(len(self.multi))])
        i = 0
        for key, value in Multifea.items():
            self.input_dim += value[2]
            self.Multi_Embed[i] = nn.Embedding(value[0], value[2], padding_idx = padding_idx)
            i+= 1
        for key, value in word2vec_fea.items():
            self.input_dim += value
        self.encoder = EncoderLayer(model_dim = 32) # 默认qkv_dim = 64, head_num = 4
###########################
        self.input_dim += int(len(self.sparse+self.multi+self.word2vec_fea)*(len(self.sparse+self.multi+self.word2vec_fea)-1)/2)*value

        # input_dim 等于 age + gender + country + city_level + video_release_date + video_score + video_duration + emb_dims
        self.hidden_dim = hidden_dim
        self.num_expert = num_expert
        self.tasks_dim = tasks_dim

        self.device = device

        ###### by Kang
        self.padding_idx = padding_idx
        ######

        self.expert = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim*num_expert),
            nn.ReLU(),
            nn.Linear(hidden_dim*num_expert, hidden_dim*num_expert),
            nn.ReLU()
        )
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, num_expert*len(tasks_dim)),
            nn.ReLU()
        )
        # task1
        self.task1_outputLayer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tasks_dim[0])
        )

        self.task2_outputLayer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tasks_dim[1])
        )
    def forward(self, input):
        # input  = [batch_size, ]
        inputs = torch.tensor(input[self.dense].values, dtype = torch.float32).to(self.device)   #input用df储存


        ######## By Kang
        encoder_inputs,mask = None,None
        for i in range(len(self.sparse)):
            name = self.sparse[i]
            emb = self.Embed[i](torch.tensor(input[name].values, dtype = torch.long).to(self.device))
            if encoder_inputs == None:
                encoder_inputs = emb.unsqueeze(1) # [batch_size, 1, 32]
            else:
                encoder_inputs = torch.cat((encoder_inputs, emb.unsqueeze(1)), dim = 1) # 沿着seq_len的维度拼接
        # 产生mask矩阵
        # mask = torch.zeros([self.encoder_input_length, encoder_inputs.shape[1]]).to(self.device) # [总的seq_len, Sparase特征序列长度], Sparase特征不需要mask
        # mask = mask.repeat(encoder_inputs.shape[0], 1, 1) # [batch_size, 总的seq_len, Sparase特征序列长度]
        for i in range(len(self.multi)):
            name = self.multi[i]
            fea_input = torch.tensor(input[name].to_list(), dtype = torch.long).to(self.device) # [batch_size, seq_len]
            emb = self.Multi_Embed[i](fea_input) # [batch_size, seq_len = 9, 32]
            emb = emb.sum(dim=1).unsqueeze(1)    #注意改前面 input_dim 大小
            encoder_inputs = torch.cat((encoder_inputs, emb), dim=1)  # 沿着seq_len的维度拼接
            # 拼接mask矩阵
            # fea_mask = (fea_input == self.padding_idx).to(self.device)[:, None, :] # [batch_size, 1, 此muti特征的序列长度], 可能需要将bool类型转化为int或者Long类型
            # fea_mask = fea_mask.repeat(1, self.encoder_input_length, 1) # [batch_size, 总的seq_len, 此muti特征的序列长度]
            # mask = torch.cat((mask, fea_mask), dim = -1) # [batch_size, 总的seq_len, 之前的序列长度+此muti特征的序列长度], 沿着最后一个维度拼接
        for i in range(len(self.word2vec_fea)):
            name = self.word2vec_fea[i]
            emb = torch.tensor(input[name].to_list(), dtype = torch.float32).to(self.device)
            emb = emb.unsqueeze(1)
            encoder_inputs = torch.cat((encoder_inputs, emb), dim=1)
            # mask = torch.cat((mask, torch.zeros([encoder_inputs.shape[0], self.encoder_input_length, 1]).to(self.device)), dim = -1)
        
#可加一个FM的特征交叉, 注意改input维度
        for i in range(len(self.sparse+self.multi+self.word2vec_fea)-1):
            for j in range(i+1,len(self.sparse+self.multi+self.word2vec_fea)):
                new = encoder_inputs[:,i,:]*encoder_inputs[:,j,:]
                encoder_inputs = torch.cat((encoder_inputs,new.unsqueeze(1)),dim=1)
        
        # mask = mask.unsqueeze(1) # 增加一个多头的维度
        # 按照txt中的方法，encoder_inputs应为[batch_size, 33, 32], 对应mask矩阵[batch_size, 33, 33]
        encoder_outputs = self.encoder(encoder_inputs, mask) # [batch_size, 9, 32]
        encoder_outputs = encoder_outputs.reshape([encoder_outputs.shape[0], encoder_outputs.shape[1]*encoder_outputs.shape[2]]) # [batch_size, 33*32]
        inputs = torch.cat((inputs, encoder_outputs), dim = -1) # [batch_size, self.input_dim = 1098]
        inputs= inputs.float()
        ######### end by Kang

        expert_output = self.expert(inputs) # [batch_size, hidden_dim*num_expert]
        expert_output = expert_output.reshape([inputs.shape[0], self.num_expert, self.hidden_dim])  # # [batch_size, num_expert, hidden_dim]
        score = self.gate(inputs) # [batch_size, num_expert*2]
        score = score.reshape([inputs.shape[0], len(self.tasks_dim), self.num_expert]) # [batch_size, 2, num_expert]
        score = nn.functional.softmax(score, dim = -1)

        tower_input = torch.bmm(score, expert_output) # [batch_size, 2, hidden_dim]
        task1_output = self.task1_outputLayer(tower_input[:, 0, :])
        task2_output = self.task2_outputLayer(tower_input[:, 1, :])
        task1_output = nn.functional.softmax(task1_output, dim = -1)
        task2_output = nn.functional.softmax(task2_output, dim = -1)
        return task1_output,task2_output

task2_loss = focal_loss(alpha=0.5, gamma=2)
def multi_focal(preds,labels,alpha=0.5, gama=2, num=10):
    label=torch.unsqueeze(labels,dim=1)
    one_hot = (0*preds).scatter_(1, label, 1).type(torch.int64)
    upred = 1-preds
    loss_fn = focal_loss(alpha,gama)
    nloss=0
    for i in range(num):
        nloss += 0.2*i*loss_fn(torch.cat((upred[:,i].view(-1,1),preds[:,i].view(-1,1)),axis=1),one_hot[:,i])
    return nloss
def watch_onehot(x,classn=10):
    a=torch.zeros((x.shape[0],classn))  #默认10分类
    for i in range(x.shape[0]):
        a[i,int(x[i])]=1
    return a
def multi_PRC(label, pre):   #label是一维，pre是n维
    Prec1, Recall1 = 0,0
    for i in range(1,pre.shape[1]): #0标签不算进去
        t,y = label[:,i],pre[:,i]
        Prec1 += 0.2*i*precision_score(t,y)
        Recall1 += 0.2*i*recall_score(t,y)
    return Prec1/i, Recall1/i
def evaluat(val_data,batch_size=512):
    loss,siters = 0, val_data.shape[0]//batch_size+1
    t1,y1,t2,y2=np.array([]),np.array([]),np.array([]),np.array([])
    with torch.no_grad():
        for i in tqdm.tqdm(range(siters)):
            batch=val_data.iloc[i*batch_size:(i+1)*batch_size]
            is_share, watch_label = torch.tensor(batch['is_share'].values,dtype=torch.int64).to(device), torch.tensor(batch['watch_label'].values,dtype=torch.int64).to(device)
            task1_output, task2_output = mmoe(batch[Dense_fea+list(Sparse_fea.keys())+list(multi_fea.keys())+list(embed_fea.keys())])
            loss += multi_focal(task1_output, watch_label).cpu().item() + task2_loss(task2_output, is_share).cpu().item()
            task2_output = task2_output[:, 1] # 取输出是1的概率计算
            if i==0:y1= task1_output.detach().cpu().numpy()
            else:y1 = np.concatenate((y1,task1_output.detach().cpu().numpy()),axis=0)
            t1 = np.concatenate((t1,watch_label.cpu().numpy()))
            t2 ,y2 = np.concatenate((t2,is_share.cpu().numpy())), np.concatenate((y2,task2_output.detach().cpu().numpy()))
    y1,y2 = watch_getLabel(y1),share_getLabel(y2)   #转换后耗时增加
    t1,y1 = watch_onehot(t1),watch_onehot(y1)
    # t1 = watch_onehot(t1)
    # Prec1, Recall1 ,Prec2, Recall2 =0,0,0,0
    uAUC = evaluation(y1,t1,y2,t2)
    Prec2, Recall2 = precision_score(t2,y2),recall_score(t2,y2)
    Prec1, Recall1 = multi_PRC(t1,y1)
    print("val_uAUC:",uAUC, "val_loss:", loss/siters, "val_PRC1",Prec1, Recall1,  "val_PRC2",Prec2, Recall2 )
    return uAUC, loss/siters, Prec1, Recall1 ,Prec2, Recall2 
def evaluation(y1,t1,y2,t2):
    score=0
    for i in range(1,10):
        T,Y = t1[:,i], y1[:,i]
        if T.sum().item()==0: auc0 = 1-Y.mean().item()   #全零则计算平均度作为AUC
        else: auc0 = roc_auc_score(T,Y)
        score += 0.1*i*auc0
    T,Y = np.array(t2), np.array(y2)
    if T.sum()==0:auc0 = 1-Y.mean()   #全零则计算平均度作为AUC
    else: auc0 = roc_auc_score(T,Y)
    print("watch_label(4.5):",score,"is_share(1):",auc0)
    return 0.7*score+0.3*auc0
def watch_getLabel(x):    #与一般不同，需要拆分成10个二分类，然后每列取 max*0.5 为阈值 归一化后再横向比较
    a=np.zeros((x.shape[0],))
    b=np.zeros((x.shape[0],x.shape[1]))
    w=[1,0.5]+[0.6]*7+[0.5]
    for i in range(x.shape[1]):    #尽可能少的预测正确0标签
        xmax,xmin=np.max(x[:,i]),np.min(x[:,i])
        for j in range(x.shape[0]):
            if x[j,i]>w[i]*(xmax-xmin)+xmin:b[j,i]=1
    #再横向比较最大值
    c=np.array(x)*b
    for i in range(x.shape[0]):
        a[i]=np.argmax(c[i,:])
    return a
def share_getLabel(x):    #按 0.5*np.max(x) 为阈值 归一化后来定论
    a=np.zeros((len(x),))
    xmax,xmin=np.max(x),np.min(x)
    for i in range(len(x)):
        if x[i]>0.6*(xmax-xmin)+xmin:a[i]=1
    return a
def submit(test_data,batch_size=512):
    submits=pd.read_csv('./testdata/test.csv')
    siters=test_data.shape[0]//batch_size+1
    watchlist,sharelist=np.array([]),np.array([])
    with torch.no_grad():
        for i in tqdm.tqdm(range(siters)):
            batch=test_data.iloc[i*batch_size:(i+1)*batch_size]
            task1_output, task2_output = mmoe(batch[Dense_fea+list(Sparse_fea.keys())+list(multi_fea.keys())+list(embed_fea.keys())])
            task2_output = task2_output[:, 1] # 取输出是1的概率计算交叉熵loss
            task1_output,task2_output = task1_output.detach().cpu().numpy(), task2_output.detach().cpu().numpy()
            if i==0:watchlist = task1_output
            else:watchlist = np.concatenate((watchlist, task1_output),axis=0)
            sharelist = np.concatenate((sharelist, task2_output),axis=0)
    np.save("watch0.npy",watchlist)
    np.save("share0.npy",sharelist)
    watchlist = watch_getLabel(watchlist).tolist()
    sharelist = share_getLabel(sharelist).tolist()
    print("length check:",len(watchlist),submits.shape[0])
    submits['watch_label'] = watchlist
    submits['is_share'] = sharelist
    submits['watch_label'] = submits['watch_label'].astype(np.int64)
    submits['is_share'] = submits['is_share'].astype(np.int64)
    print(sub_process(submits))
    submits.to_csv("submits.csv",index=False)
def sub_process(a):
    print("分享（正/all）:",(a['is_share']==1).sum()/a.shape[0])
    print("看了（0/all）:",(a['watch_label']==0).sum()/a.shape[0])
    print("看了（1/all）:",(a['watch_label']==1).sum()/a.shape[0])
    print("看了（2/all）:",(a['watch_label']==2).sum()/a.shape[0])
    print("看了（3/all）:",(a['watch_label']==3).sum()/a.shape[0])
    print("看了（4/all）:",(a['watch_label']==4).sum()/a.shape[0])
    print("看了（5/all）:",(a['watch_label']==5).sum()/a.shape[0])
    print("看了（6/all）:",(a['watch_label']==6).sum()/a.shape[0])
    print("看了（7/all）:",(a['watch_label']==7).sum()/a.shape[0])
    print("看了（8/all）:",(a['watch_label']==8).sum()/a.shape[0])
    print("看了（9/all）:",(a['watch_label']==9).sum()/a.shape[0])
def fill(data):
    for name in list(multi_fea.keys()):
        error=data[name].isin(['0'])
        data.loc[error,name]="["+"0,"*8+"0"+"]"
        data[name]=data[name].apply(lambda x:[int(i) for i in x.strip('[').strip(']').split(',')])
    for name in list(embed_fea.keys()):
        error=data[name].isin(['0'])
        data.loc[error,name]="["+"0,"*31+"0"+"]"
        data[name]=data[name].apply(lambda x:[float(i) for i in x.strip('[').strip(']').split(',')])
    return data
if __name__ == '__main__':
    start_time=time.time()
    from  torch.utils.data import DataLoader
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    batch_size = 256
    lr = 0.0001
    decay = 0.1
    totalEpoch = 20
    valratio = 0.2

    hidden_dim = 256
    tasks_dim = [10, 2]
    names = ['user_id','video_id']
    newlabel=[f"{names[0][:5]}label{j}" for j in range(6)] + [f"{names[1][:5]}label{j}" for j in range(6*len(ACTION_LIST)-14)]
    Dense_fea += [f"age{j}" for j in range(8)]+[f"gender{j}" for j in range(4)]+[f"country{j}" for j in range(3)]+[f"city_level{j}" for j in range(8)]+newlabel

    mmoe = MMoE(Dense_fea,Sparse_fea, multi_fea, embed_fea,hidden_dim, tasks_dim, device)
    mmoe.to(device)
    # for name,param in mmoe.named_parameters():
    #     if param.requires_grad==True:print(name)

    data = pd.read_pickle("./traindata/train_data.pickle").sort_values('pt_d')
    print(data.shape,data.columns)

    splitnum = int(data.shape[0]*valratio)+1
    train_data,val_data = data.iloc[:-splitnum],data.iloc[-splitnum:]
    sub_process(train_data)
    sub_process(val_data)
    iters=train_data.shape[0]//batch_size+1
    val_uAUC, val_loss, val_Prec1, val_Recall1, val_Prec2, val_Recall2 = [],[],[],[],[],[]
    t_uAUC, t_loss = [],[]
    for epoch in range(totalEpoch):
        opt = torch.optim.Adam(mmoe.parameters(), lr = lr)
        lr=lr*(1-decay)
        for i in tqdm.tqdm(range(iters)):
            batch=train_data.iloc[i*batch_size:(i+1)*batch_size]
            is_share, watch_label = torch.tensor(batch['is_share'].values,dtype=torch.int64).to(device), torch.tensor(batch['watch_label'].values,dtype=torch.int64).to(device)

            task1_output, task2_output = mmoe(batch[Dense_fea+list(Sparse_fea.keys())+list(multi_fea.keys())+list(embed_fea.keys())])
            loss =  multi_focal(task1_output, watch_label) + task2_loss(task2_output, is_share) +0.1*F.cross_entropy(task1_output, watch_label) 
            opt.zero_grad()
            loss.backward()
            opt.step()
            y1,t1,y2,t2=task1_output.detach().cpu().numpy(), watch_label.cpu().numpy(),task2_output[:,1].detach().cpu().numpy(),is_share.cpu().numpy()
            torch.cuda.empty_cache()
            # if i % 300 == 0 and i > 0:break
        if valratio==0:uauc, vloss, Prec1, Recall1 ,Prec2, Recall2  = 0,0,0,0,0,0
        else:uauc, vloss, Prec1, Recall1 ,Prec2, Recall2  = evaluat(val_data,512)
        val_uAUC, val_loss, val_Prec1, val_Recall1, val_Prec2, val_Recall2 = val_uAUC+[uauc], val_loss+[vloss], val_Prec1+[Prec1], val_Recall1+[Recall1], val_Prec2+[Prec2], val_Recall2+[Recall2] 
        print("epoch:", epoch, " loss:", loss.cpu().item())
        t_loss = t_loss+[loss.cpu().item()]
    
    del data,train_data,val_data,batch
    gc.collect()
    print("耗时（min）：",(time.time()-start_time)/60)
    import matplotlib.pyplot as plt
    plt.subplot(2,3,1)
    plt.plot(range(1,totalEpoch+1),val_uAUC)
    plt.title('uAUC')
    plt.subplot(2,3,2)
    plt.plot(range(1,totalEpoch+1),t_loss)
    plt.plot(range(1,totalEpoch+1),val_loss)
    plt.legend(['train','val'])
    plt.title('loss')
    plt.subplot(2,3,3)
    plt.plot(range(1,totalEpoch+1),val_Prec1)
    plt.title('val_Prec1')
    plt.subplot(2,3,4)
    plt.plot(range(1,totalEpoch+1),val_Recall1)
    plt.title('val_Recall1')
    plt.subplot(2,3,5)
    plt.plot(range(1,totalEpoch+1),val_Prec2)
    plt.title('val_Prec2')
    plt.subplot(2,3,6)
    plt.plot(range(1,totalEpoch+1),val_Recall2)
    plt.title('val_Recall2')
    # plt.show()
    
    test_data=pd.read_pickle("./traindata/test_data.pickle")
    submit(test_data)
    print("耗时（min）：",(time.time()-start_time)/60)
    print(val_uAUC,'\n', val_loss,'\n', val_Prec1,'\n', val_Recall1,'\n', val_Prec2,'\n', val_Recall2)
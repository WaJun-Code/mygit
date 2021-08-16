# -*- coding: utf-8 -*-
import time
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import re
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()

def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_b.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

def evaluate(model, x, y,user_id_list):
    pred_ans = model.predict(x, verbose=1, batch_size=1024)
    return uAUC(y,pred_ans,user_id_list)
def get_hist(userFA,test,names,startday=6):  #，获取ufa对应label统计信息（曝光率之类），并以label0-15记之,对稠密特征统计
    userFA['RLCF']=userFA[["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]].sum(axis=1)
    train_df=userFA[userFA['RLCF']!=0]
    for i in range(14-startday+2):
        train=userFA[userFA['date_']==i+startday][names+['date_']]
        for name in names:
            a=train_df[(train_df["date_"] > i) & (train_df["date_"] < i+startday)][[name,'feedid']].groupby(name).agg({list})
            b=a.index.to_frame().reset_index(drop=True)
            a=pd.concat((b,pd.DataFrame(columns=['hist_feedid'],data=a.values)),axis=1)  #两列：userid,hist_feeid，某天之前的user
            if i+startday<15:
                train=pd.merge(train,a,on=name,how='left').fillna(0)  #一共7个train，三列：userid，date,hist_feeid
            else:test=pd.merge(test,a,on=name,how='left').fillna(0)
        if i==0:c=train.copy()
        if 0<i<15-startday:c=pd.concat((c,train)).reset_index(drop=True)
    return c,test
def cos_sim(a, b):    #计算余弦相似度函数
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if b_norm==0:return 0    #只有userEmb可能为0
    else:return np.dot(a,b)/(a_norm * b_norm)
def get_recall(userid,userEmb,feed_embed,n):
    userEmbI=pd.DataFrame(index=userEmb['userid'],data=userEmb.iloc[:,1:513].values)
    feedemb=feed_embed.copy()
    # feedemb['sim']=feedemb['feed_embedding'].parallel_apply(lambda x:cos_sim(x,userEmbI.loc[userid]))
    feedemb['sim']=feedemb['feed_embedding'].apply(lambda x:cos_sim(x,userEmbI.loc[userid]))
    feedemb=feedemb.sort_values('sim').reset_index(drop=True)
    return feedemb['feedid'].tolist()[-n:]

# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_feature_columns, behavior_fea, his_behavior_fea, emb_dim=8, max_len=100):

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_fea]
    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=112872,
                                    embedding_dim=emb_dim, embedding_name='feedid'), maxlen=max_len) for feat in his_behavior_fea]
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in his_behavior_fea:
            # 这是历史行为序列
            his_list = [json.loads(l) for l in df[name]]   #直接文件里读取list时，需要json操作
            for i in range(len(his_list)):
                if his_list[i]==0:his_list[i]=[0]    #缺失值处理
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')      # hist三维数组
        else:
            x[name] = df[name].values
    return x, dnn_feature_columns

if __name__ == "__main__":
    t=time.time()
    submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
    score=0

    feed_info_df = pd.read_csv(FEED_INFO).fillna(0)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"]].fillna(0)
    origUser=user_action_df['userid'].values.copy()
    user_action_df['userid'] = lbe.fit_transform(user_action_df['userid'])   #先对userid进行lbe
    lbeUser=pd.DataFrame(index=origUser,data=user_action_df['userid'].values).drop_duplicates(keep='last')
    userFA=pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left').fillna(0)

    name=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list','machine_tag_list','description']
    wordcol=[f"word{j}"for j in range(8*len(name))]
    #此处改embedding维度
    columns,usercol,authorcol=[f"embed{i}" for i in range(64)],[f"uembed{i}" for i in range(8)],[f"aembed{i}" for i in range(8)]
    #自创特征与数据预处理处不同：
    labelCol=[f"ulabel{j}" for j in range(16)]+[f"flabel{j}" for j in range(16)]+[f"alabel{j}" for j in range(16)]

    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    feed_embed['feed_embedding']=feed_embed['feed_embedding'].apply(lambda x:np.array([float(i) for i in str(x).strip().split(" ")]))

    for action in ACTION_LIST:

        target = [action]
        data = pd.read_csv(ROOT_PATH + f'/train_test_data_for_{action}.csv')

        # 对userEmb的获取
        # userEmb=pd.read_csv("userEmb.csv",index_col=0)
        # a=np.zeros([20000-userEmb.shape[0],513])
        # a[:,0]=[i for i in lbeUser.index.tolist() if i not in userEmb.index.tolist()]
        # usercol=['userid']+[f"uembed{i}" for i in range(512)]
        # a=pd.DataFrame(columns=usercol,data=a)
        # userEmb=pd.read_csv("userEmb.csv",header=None,names=usercol).drop(0)
        # userEmb=pd.concat((userEmb,a),axis=0).reset_index(drop=True)
        # #对authorEmb的获取
        # authorcol=['authorid']+[f"aembed{i}" for i in range(512)]
        # authorEmb=pd.read_csv("authorEmb.csv",header=None,names=authorcol).drop(0)
        # data['hist_recall']= data['userid'].apply(lambda x:get_recall(x,userEmb,feed_embed,256))  #召回256个作为历史序列

        # columns=[]
        # usercol,authorcol=[],[]
        # labelCol=[]
        
        if action=='forward':epochs=2
        elif action=='click_avatar':epochs=2
        else:epochs=1

        # USE_FEAT= ['userid','hist_feedid','hist_recall', action] + FEA_FEED_LIST +labelCol+columns+usercol+authorcol+wordcol
        USE_FEAT= ['userid','hist_feedid', action] + FEA_FEED_LIST +labelCol+columns+usercol+authorcol+wordcol  #有无hist_recall
        dense_features = ['videoplayseconds']+labelCol+columns+usercol+authorcol+wordcol

        # behavior_fea ,hist_behavior_fea= ['feedid'], ['hist_feedid','hist_recall']         #通过groupby+agg做hist特征
        behavior_fea ,hist_behavior_fea= ['feedid'], ['hist_feedid']         #通过groupby+agg做hist特征  
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target+hist_behavior_fea]
        sparseNum={}
        for feat in sparse_features:
            sparseNum[feat]=data[feat].nunique()
            if feat=='userid':sparseNum[feat]=20000
            elif feat=='feedid':sparseNum[feat]=112872
            elif feat=='authorid':sparseNum[feat]=18789
        sparse_feature_columns = [SparseFeat(feat, sparseNum[feat], embedding_dim=8) for feat in sparse_features]

        train,test= data.iloc[:-submit.shape[0]].reset_index(drop=True), data.iloc[-submit.shape[0]:].reset_index(drop=True)

        # c,test= get_hist(userFA,test,['userid'])   #lbe之前做hist特征
        # c = c.drop_duplicates(['userid', 'date_'], keep='last')
        # train= pd.merge(train, c, on=['userid','date_'], how='left').fillna(0)
        #控制负样本比例
        # train=pd.concat((train[train[action]==0].sample(frac=0.8,random_state=0,replace=False),train[train[action]==1]))

        train = train.sort_values('date_')   #按照日期排序，防止数据泄露
        train = train[USE_FEAT]

        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])

        test = test[[i for i in USE_FEAT if i != action]]
        test[target[0]] = 0
        test = test[USE_FEAT].fillna(0)
        num=int((1-0.25)*train.shape[0])

        x_train,x_valid=train.iloc[:num],train.iloc[num:]
        x_train,x_valid=train,train.iloc[num:]    #此处调节是否留验证集
        x_trn, dnn_feature_columns = get_din_feats_columns(x_train, dense_features, sparse_feature_columns, behavior_fea, hist_behavior_fea, max_len=50)
        y_trn = x_train[target].values
        x_val, dnn_feature_columns = get_din_feats_columns(x_valid, dense_features, sparse_feature_columns, behavior_fea, hist_behavior_fea, max_len=50)
        y_val = x_valid[target].values

        x_tst, dnn_feature_columns = get_din_feats_columns(test, dense_features, sparse_feature_columns, behavior_fea, hist_behavior_fea, max_len=50)

        model = DIN(dnn_feature_columns, behavior_fea)
        model.compile('adam', 'binary_crossentropy',metrics=['binary_crossentropy', tf.keras.metrics.AUC()])
        history = model.fit(x_trn, y_trn, verbose=1, epochs=epochs, validation_data=(x_val, y_val) , batch_size=2048)
        
        user_id_list=x_valid['userid'].values
        action_auc=evaluate(model,x_val,y_val,user_id_list)
        print(action,action_auc)
        #计算验证集上的score
        if action=='read_comment':
            score+=0.4*action_auc
        elif action=='like':
            score+=0.3*action_auc
        elif action=='click_avatar':
            score+=0.2*action_auc
        else:
            score+=0.1*action_auc

        pred_ans = model.predict(x_tst, verbose=1, batch_size=1024)
        print("长度确认：",len(pred_ans)==submit.shape[0])
        submit[action] = pred_ans
    submit.to_csv("./submit_base_DIN.csv", index=False)
    print('Time cost: %.2f s'%(time.time()-t))
    print("验证集上得分score=",score)
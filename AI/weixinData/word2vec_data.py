# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import re
def cut(x):
    if x==0:return 0
    y=[]
    y0=re.split(' |;',str(x).strip())
    try:
        for i in range(len(y0)):
            y.append(str(y0[i]))
    except:
        y=0
    return y
def strcut(x):
    if x==0:return 0
    y=[]
    y0=re.split(' |;',str(x).strip())
    try:
        for i in range(len(y0)):
            if (i+1)%2==0:
                if float(y0[i])>0.01:y.append(str(y0[i-1]))
    except:
        y=0
    return y
def get_dfcut(x):
    y=np.zeros([8,])
    if x!=0:
        for i in x:
            y+=w2v[i]
        y=y/len(x)
    return y
def get_word(feed_info_df,name):
    y=[]
    for k in range(feed_info_df.shape[0]):
        yy=[]
        for i in range(len(name)):
            yy+=feed_info_df.iloc[k][name[i]].tolist()
        y.append(yy)
    a=pd.DataFrame(columns=[f"word{j}"for j in range(8*len(name))],data=y)
    feed_info_df=feed_info_df.drop(columns=name,axis=1)
    feed_info_df=pd.concat((feed_info_df,a),axis=1)
    return feed_info_df
# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(1/10比例负样本)
ACTION_SAMPLE_RATE = {"read_comment": 0.24, "like": 0.2, "click_avatar": 0.07, "forward": 0.035, "comment": 0.1, "follow": 0.1,
                      "favorite": 0.1}
t=time.time()
feed_info_df = pd.read_csv(FEED_INFO).fillna(0)
user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST].fillna(0)
feed_embed = pd.read_csv(FEED_EMBEDDINGS)
test = pd.read_csv(TEST_FILE)[['userid','feedid']]
index=(user_action_df['feedid'].apply(lambda x:x not in feed_info_df['feedid'] and x not in feed_embed['feedid']))
user_action_df=user_action_df.drop(user_action_df[index].index)   #去掉feedid和emb不在feedinfo里的436610个数据

doc1=feed_info_df[feed_info_df['manual_keyword_list']!=0]['manual_keyword_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
doc2=feed_info_df[feed_info_df['machine_keyword_list']!=0]['machine_keyword_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
doc3=feed_info_df[feed_info_df['manual_tag_list']!=0]['manual_tag_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
doc4=feed_info_df[feed_info_df['machine_tag_list']!=0]['machine_tag_list'].apply(strcut).tolist()
doc5=feed_info_df[feed_info_df['description']!=0]['description'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
from gensim.models import Word2Vec
docs = doc1+doc2+doc3+doc4+doc5
w2v = Word2Vec(docs, size=8, sg=1, window=3, seed=2020, workers=2, min_count=1, iter=10)

feed_info_df['machine_tag_list']=feed_info_df['machine_tag_list'].apply(strcut).apply(get_dfcut)
feed_info_df['manual_tag_list']=feed_info_df['manual_tag_list'].apply(cut).apply(get_dfcut)
feed_info_df['manual_keyword_list']=feed_info_df['manual_keyword_list'].apply(cut).apply(get_dfcut)
feed_info_df['machine_keyword_list']=feed_info_df['machine_keyword_list'].apply(cut).apply(get_dfcut)
feed_info_df['description']=feed_info_df['description'].apply(cut).apply(get_dfcut)
feed_info_df=get_word(feed_info_df,name)
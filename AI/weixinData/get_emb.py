# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
def meanEmb(user_feed_list,feed_emb_list):
    mean=np.zeros([512,])
    for i in user_feed_list:
        mean+=np.array(feed_emb_list.loc[i][0][0])
    return mean/len(user_feed_list)
from sklearn.cluster import KMeans
def kmeansEmb(user_feed_list,feed_emb_list):
    x,n_cluster=[],len(user_feed_list)//32+1
    kmeans=KMeans(n_clusters=n_cluster)
    for i in user_feed_list:
        x.append(feed_emb_list.loc[i][0][0])
    x=np.array(x)
    kmeans.fit(x)
    y=kmeans.predict(x)   #分类是从0开始
    ynum=[y.tolist().count(i) for i in range(n_cluster)]
    ymax=np.array(ynum).argmax()
    index=[i for i in range(len(y)) if y[i]==ymax]
    return np.mean(x[index],axis=0)
def get_UAemb(userFA,feedemb,name):
    b=userFA[[name,'feedid']].groupby(name).agg({list})
    userList=b.index.tolist()
    feed_emb=feedemb.copy()
    feed_emb['feed_embedding']=feed_emb['feed_embedding'].apply(lambda x:[float(i) for i in str(x).strip().split(" ")])
    feed_emb_list=feed_emb.groupby('feedid').agg({list})   #查询表
    userEmb=[]
    for i in range(len(userList)):
        user_feed_list=b.loc[userList[i]][0]
        if len(user_feed_list)<32:
            userEmb.append(meanEmb(user_feed_list,feed_emb_list).tolist())
        else:
            userEmb.append(kmeansEmb(user_feed_list,feed_emb_list).tolist())
    return pd.DataFrame(index=userList,data=userEmb)       #得到可通过index索引获取emb的df

#对userEmb的获取需要先把userAction处理成user里仅含正面评论的df
t=time.time()
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
feedemb=pd.read_csv(FEED_EMBEDDINGS)
feedinfo=pd.read_csv(FEED_INFO)
userAction=pd.read_csv(USER_ACTION)

user_Action=userAction[['userid','feedid']]
user_Action['RLCF']=userAction['read_comment']+userAction['like']+userAction['click_avatar']+userAction['forward']
userIn=user_Action[user_Action['RLCF']!=0]
get_UAemb(userIn,feedemb,'userid').to_csv("userEmb.csv")
print('Time cost: %.2f s'%(time.time()-t))

# userFA=pd.merge(userAction,feedinfo[['feedid','authorid','machine_keyword_list', 'manual_tag_list']],on='feedid',how='left')
# get_UAemb(userFA,feedemb,'authorid').to_csv("authorEmb.csv")
get_UAemb(feedinfo,feedemb,'authorid').to_csv("authorEmb.csv")
print('Time cost: %.2f s'%(time.time()-t))
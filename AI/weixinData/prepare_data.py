# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandarallel import pandarallel   #只有在Linux下运行
pandarallel.initialize()
from tqdm import tqdm
import time
import re
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
lbe = LabelEncoder()

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

def meanEmb(user_feed_list,feed_emb_list):
    mean=np.zeros([512,])
    for i in user_feed_list:
        mean+=np.array(feed_emb_list.loc[i][0][0])
    return mean/len(user_feed_list)
from sklearn.cluster import KMeans
def kmeansEmb(user_feed_list,feed_emb_list):
    x,n_cluster=[],1+len(user_feed_list)//32
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
# 负样本下采样比例(1/10比例负样本)
ACTION_SAMPLE_RATE = {"read_comment": 0.24, "like": 0.2, "click_avatar": 0.07, "forward": 0.035, "comment": 0.1, "follow": 0.1,
                      "favorite": 0.1}
def PCA_emb(userEmb,name):
    temp=userEmb.iloc[:,1:513]
    num=8   #此处该deepfm里也要改
    pca2=PCA(n_components=num)
    pca2.fit(temp)
    temp=pca2.transform(temp)
    userPemb=pd.DataFrame(columns=[f"{name[0]}embed{i}" for i in range(num)],data=temp)
    userPemb=pd.concat((userPemb,userEmb[name].to_frame()),axis=1)
    return userPemb
def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm_notebook(range(train.shape[0])):
        x = train.iloc[i]['feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y[:512]
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    num=64   #此处该deepfm里也要改
    pca2=PCA(n_components=num)
    pca2.fit(temp)
    temp=pca2.transform(temp)
    temp=pd.DataFrame(columns=[f"embed{i}" for i in range(num)],data=temp.tolist())
    train = pd.concat((train, temp), axis=1)
    train=train.drop('feed_embedding',axis=1)
    return train

def get_dense(userFA,test,names,startday=6):  #，获取ufa对应label统计信息（曝光率之类），并以label0-15记之,对稠密特征统计
    for i in range(14-startday+2):
        train=userFA[userFA['date_']==i+startday]
        for name in names:
            a=userFA[(userFA["date_"] > i) & (userFA["date_"] < i+startday)][[name,'videoplayseconds']+ACTION_LIST].groupby(name).agg(['mean','sum','std','count'])        
            b=a.index.to_frame().reset_index(drop=True)
            a=a.drop(columns=[(i,'count') for i in ACTION_LIST])
            a=pd.concat((b,pd.DataFrame(columns=[f"{name[0]}label{j}" for j in range(20-4)],data=a.values)),axis=1)
            if i+startday<15:
                train=pd.merge(train,a,on=name,how='left').fillna(0)  #一共7个train
            else:test=pd.merge(test,a,on=name,how='left').fillna(0)
        if i==0:c=train.copy()
        if 0<i<15-startday:c=pd.concat((c,train)).reset_index(drop=True)
    return c,test
def cos_sim(a, b):    #计算余弦相似度函数
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if b_norm==0:return 0    #只有userEmb可能为0
    else:return np.dot(a,b)/(a_norm * b_norm)

def get_sim(data,userEmb,authorEmb,feed_embed):
    userEmbI=pd.DataFrame(index=userEmb['userid'],data=userEmb.iloc[:,1:513].values)
    authorEmbI=pd.DataFrame(index=authorEmb['authorid'],data=authorEmb.iloc[:,1:513].values)
    
    feedEmbI=pd.DataFrame(index=feed_embed['feedid'],data=feed_embed['feed_embedding'].values)
    columns=[f"sim{j}"for j in range(2)]
    data[columns]=0
    for i in tqdm_notebook(range(data.shape[0])):
        userid,feedid,authorid=data.iloc[i]['userid'],data.iloc[i]['feedid'],data.iloc[i]['authorid']
        data.loc[i,columns[0]]=cos_sim(feedEmbI.loc[feedid][0],userEmbI.loc[userid])
        data.loc[i,columns[1]]=cos_sim(authorEmbI.loc[authorid],userEmbI.loc[userid])
    return data

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
def get_recall(userid,userEmb,feed_embed,n):
    userEmbI=pd.DataFrame(index=userEmb['userid'],data=userEmb.iloc[:,1:513].values)
    feedemb=feed_embed.copy()
    # feedemb['sim']=feedemb['feed_embedding'].parallel_apply(lambda x:cos_sim(x,userEmbI.loc[userid]))
    feedemb['sim']=feedemb['feed_embedding'].apply(lambda x:cos_sim(x,userEmbI.loc[userid]))
    feedemb=feedemb.sort_values('sim').reset_index(drop=True)
    return feedemb['feedid'].tolist()[-n:]
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

if __name__ == "__main__":
    t=time.time()
    feed_info_df = pd.read_csv(FEED_INFO).fillna(0)
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST].fillna(0)
    origUser=user_action_df['userid'].values.copy()
    user_action_df['userid'] = lbe.fit_transform(user_action_df['userid'])   #先对userid进行lbe
    lbeUser=pd.DataFrame(index=origUser,data=user_action_df['userid'].values).drop_duplicates(keep='last')
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    test = pd.read_csv(TEST_FILE)[['userid','feedid']]
    test['userid']=test['userid'].apply(lambda x : lbeUser.loc[x][0])
    index=(user_action_df['feedid'].apply(lambda x:x not in feed_info_df['feedid'] and x not in feed_embed['feedid']))
    user_action_df=user_action_df.drop(user_action_df[index].index)   #去掉feedid和emb不在feedinfo里的436610个数据
    userFA=pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left').fillna(0)
    
    #将feedinfo拆词
    name=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list','machine_tag_list','description']
    wordcol=[f"word{j}"for j in range(8*len(name))]
    FEA_FEED_LIST+=wordcol   #是否加入key和tag
    #做word2vec
    doc1=feed_info_df[feed_info_df['manual_keyword_list']!=0]['manual_keyword_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
    doc2=feed_info_df[feed_info_df['machine_keyword_list']!=0]['machine_keyword_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
    doc3=feed_info_df[feed_info_df['manual_tag_list']!=0]['manual_tag_list'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
    doc4=feed_info_df[feed_info_df['machine_tag_list']!=0]['machine_tag_list'].apply(strcut).tolist()
    doc5=feed_info_df[feed_info_df['description']!=0]['description'].apply(lambda x:[str(i) for i in re.split(' |;',str(x).strip())]).tolist()
    docs = doc1+doc2+doc3+doc4+doc5
    w2v = Word2Vec(docs, size=8, sg=1, window=3, seed=2020, workers=2, min_count=1, iter=10)

    feed_info_df['machine_tag_list']=feed_info_df['machine_tag_list'].apply(strcut).apply(get_dfcut)
    feed_info_df['manual_tag_list']=feed_info_df['manual_tag_list'].apply(cut).apply(get_dfcut)
    feed_info_df['manual_keyword_list']=feed_info_df['manual_keyword_list'].apply(cut).apply(get_dfcut)
    feed_info_df['machine_keyword_list']=feed_info_df['machine_keyword_list'].apply(cut).apply(get_dfcut)
    feed_info_df['description']=feed_info_df['description'].apply(cut).apply(get_dfcut)
    feed_info_df=get_word(feed_info_df,name)
    
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left').fillna(0)
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left').fillna(0)
    test["videoplayseconds"],train["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0),np.log(train["videoplayseconds"] + 1.0)

    #处理test、userAction加上统计量特征
    names=['userid','feedid','authorid']
    train,test=get_dense(train,test,names)    #添加特征函数
    labelCol=[f"ulabel{j}" for j in range(16)]+[f"flabel{j}" for j in range(16)]+[f"alabel{j}" for j in range(16)]

    train,test=pd.merge(train,feed_embed,on='feedid', how='left'),pd.merge(test,feed_embed,on='feedid', how='left')  #引入emb
    feed_embed['feed_embedding']=feed_embed['feed_embedding'].apply(lambda x:np.array([float(i) for i in str(x).strip().split(" ")]))
    c,test= get_hist(userFA,test,['userid'])   #lbe之前做hist特征
    c = c.drop_duplicates(['userid', 'date_'], keep='last')
    train= pd.merge(train, c, on=['userid','date_'], how='left').fillna(0)

    for action in tqdm(ACTION_LIST):
        # 对userEmb的获取
        userEmb=pd.read_csv("userEmb.csv",index_col=0)
        a=np.zeros([20000-userEmb.shape[0],513])
        a[:,0]=[i for i in lbeUser.index.tolist() if i not in userEmb.index.tolist()]
        usercol=['userid']+[f"uembed{i}" for i in range(512)]
        a=pd.DataFrame(columns=usercol,data=a)
        userEmb=pd.read_csv("userEmb.csv",header=None,names=usercol).drop(0)
        userEmb=pd.concat((userEmb,a),axis=0).reset_index(drop=True)
        #对authorEmb的获取
        authorcol=['authorid']+[f"aembed{i}" for i in range(512)]
        authorEmb=pd.read_csv("authorEmb.csv",header=None,names=authorcol).drop(0)

        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        # tmp = train
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all = df_all.fillna(0)
        print(action,"检查负样本比例：",df_neg.shape[0]/df_all.shape[0])

        data = pd.concat((df_all, test)).reset_index(drop=True)

        #lbe之前创建相似度特征
        # data = get_sim(data,userEmb,authorEmb,feed_embed)   #不用归一化
        print("完成相似度计算")
        # data['hist_recall']= data['userid'].apply(lambda x:get_recall(x,userEmb,feed_embed,256))  #召回256个作为历史序列
        print("完成召回计算：",data.columns)

        print('Time cost: %.2f s'%(time.time()-t))
        data=process_embed(data)
        #进行PCA降维
        userEmb,authorEmb=PCA_emb(userEmb,'userid'),PCA_emb(authorEmb,'authorid')

        data=pd.merge(data,userEmb,on='userid', how='left').fillna(0)       #引入userid和authorid的emb，需在lbe之前
        data=pd.merge(data,authorEmb,on='authorid', how='left').fillna(0)

        # add feed feature
        print("NaN判断",data.isnull().sum().sum())

        for feat in ['bgm_song_id', 'bgm_singer_id']:   #先做lbe和归一化，注意train和test必须一起做'userid','feedid', 'authorid',
            data[feat] = lbe.fit_transform(data[feat])
            print("检查类别数量：",data[feat].nunique())
        mms = MinMaxScaler(feature_range=(0, 1))
        denseFea=['videoplayseconds']+labelCol
        data[denseFea] = mms.fit_transform(data[denseFea])
        print(f"prepare data for {action}")

        data.fillna(0).to_csv(ROOT_PATH + f'/train_test_data_for_{action}.csv', index=False)
        print('Time cost: %.2f s'%(time.time()-t))
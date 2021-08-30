import pandas as pd
import tqdm
import numpy as np
from collections import defaultdict
import glob,time
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
# from pandarallel import pandarallel   #只有在Linux下运行
# pandarallel.initialize()
lbe = LabelEncoder()
mms = MinMaxScaler(feature_range=(0, 1))
##### tf-idf
# from sklearn import feature_extraction # 导入sklearn库, 以获取文本的tf-idf值
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
#
# txt = ["我也 和 你们", "机值 和 我也"]
# mat = CountVectorizer()
# tf = TfidfTransformer()
# y = mat.fit_transform(txt)
# x = mat.vocabulary_
# tfidf = tf.fit_transform(y)
# word = mat.get_feature_names()  # 单词的名称
# weight = tfidf.toarray()  # 权重矩阵, 在此示范中矩阵为(1, n)
## 属性video_release_date 直接分割成 年 月 日 三个属性
ACTION_LIST=['is_share','is_watch',  'is_collect', 'is_comment','watch_label','age', 'city_level']
Dense_fea=['age', 'gender', 'country', 'city_level','video_score', 'video_duration','video_release_date']  #gender 3种和 age与城市等级7种（不能fillna0） 可以one-hot后粘结
Sparse_fea=[ 'province', 'city' , 'device_name','video_id', 'video_name']
multi_fea=['video_director_list', 'video_actor_list','video_second_class']
multinum=9
embed_fea=['video_tags', 'video_description']
def date_separate(x):
    y=''
    if not pd.isna(x):
        for i in x.split('-'):
            y+=i
    else:y='0'
    return int(y)
def separate(x):
    if not pd.isna(x):x=x.split(',')
    else:x=['0']
    if len(x)>multinum:x=x[:multinum]   #截取前9个关键词汇
    return x
def LabelEncode(df,column):
    def label(x):
        a=[0]*multinum   #9个补零
        for i in range(len(x)):
            a[i]=cdf.loc[x[i]].values[0]
        return str(a)
    def get_vidas(x):
        num=0
        for i in x:
            if i!='0':num +=cdf0.loc[i].values[0]
        return num
    
    clist=[]
    for i in df[column]:clist+=i
    cdf0=pd.DataFrame(columns=[column],data=clist)
    cdf0 = cdf0[column].value_counts().to_frame()
    encode = lbe.fit_transform(cdf0.index)
    print(column,cdf0.shape[0])
    cdf=pd.DataFrame(index=cdf0.index,columns=[column],data=encode)
    df[column+'0']=df[column]
    df[column]=df[column].apply(label)
    df[column+'0']=df[column+'0'].apply(get_vidas)   #也可在此统计导演，演员参演电影数目
    return df

def get_dense(userFA,test,names,startday=6):  #startday==0表示第一天,获取ufa对应label统计信息（曝光率之类），并以label0-15记之,对稠密特征统计
    date_list=[20210419,20210420,20210421,20210422,20210423,20210424,20210425,20210426,20210427,20210428,20210429,20210430,20210501,20210502,20210503]
    for i in range(startday,15):  #i==14时为test，train上最大为13
        for name in names:
            if name=='video_id':
                a=userFA[userFA["pt_d"] < date_list[i]][[name]+ACTION_LIST].groupby(name).agg(['mean','min','max','sum','std','count'])
                b=a.index.to_frame().reset_index(drop=True)
                dropcol=[(j,'min') for j in ACTION_LIST[:4]]+[(j,'max') for j in ACTION_LIST[:4]]+[(j,'count') for j in ACTION_LIST[:6]]   #此处丢掉了2*4+6个
                a=a.drop(columns=dropcol)
                column=[f"{name[:5]}label{j}" for j in range(6*len(ACTION_LIST)-14)]
                a=pd.concat((b,pd.DataFrame(columns=column,data=a.values)),axis=1)
                a[column]=a[column].astype(np.float32)
            else:
                a=userFA[userFA["pt_d"] < date_list[i]][[name]+['video_score', 'video_duration','video_release_date']].groupby(name).agg(['mean','std'])
                b=a.index.to_frame().reset_index(drop=True)
                column=[f"{name[:5]}label{j}" for j in range(6)]
                a=pd.concat((b,pd.DataFrame(columns=column,data=a.values)),axis=1)
                a[column]=a[column].astype(np.float32)
            if i<14:
                train=userFA[userFA['pt_d']==date_list[i]]
                train=pd.merge(train,a,on=name,how='left')
                if i==startday:c=train.copy()
                else:c=pd.concat((c,train)).reset_index(drop=True)
            else:test=pd.merge(test,a,on=name,how='left')            
        print("完成第:",i+1,"天统计")
    return c,test
def sepSparse(df,name):
    num = df[name].nunique()
    df_0=np.zeros([df.shape[0],num])
    for i in range(df.shape[0]):df_0[i,int(df[name][i])]=1
    df_0=pd.DataFrame(columns= [f"{name}{j}" for j in range(num)], data=df_0)
    df_0[[f"{name}{j}" for j in range(num)]] = df_0[[f"{name}{j}" for j in range(num)]].astype(np.int64)
    df = pd.concat((df,df_0),axis=1)
    if name in ['gender', 'country']:df=df.drop(columns=name)
    return df,[f"{name}{j}" for j in range(num)]
def fill(data):
    for name in multi_fea:
        error=data[name].isin([0])
        data.loc[error,name]=str([0]*multinum)
        data[name] = data[name].apply(lambda x:[float(i) for i in x.strip('[').strip(']').split(',')])
    for name in embed_fea:
        error=data[name].isin([0])
        data.loc[error,name]=str([0]*32)
        data[name] = data[name].apply(lambda x:[float(i) for i in x.strip('[').strip(']').split(',')])
    return data
if __name__ == "__main__":
    start_time=time.time()
    user_feature_names = {'user_id', 'age', 'gender', 'country', 'province', 'city', 'city_level',
           'device_name'}
    video_feature_names = {'video_id', 'video_name', 'video_tags', 'video_description',
           'video_release_date', 'video_director_list', 'video_actor_list',
           'video_score', 'video_second_class', 'video_duration'}
    history_feature_names = {'user_id', 'video_id', 'is_watch', 'is_share', 'is_collect',
           'is_comment', 'watch_start_time', 'watch_label', 'pt_d'}
        
    user_features_path = "./traindata/user_features_data.csv"
    video_features_path = "./traindata/video_features_data.csv"
    history_behavior_root = "./traindata/history_behavior_data/"
    
    user_data = pd.read_csv(user_features_path, delimiter='\t', low_memory=True)
    video_data = pd.read_csv(video_features_path, delimiter='\t', low_memory=True)
    test_data=pd.read_csv('./testdata/test.csv')
    print(video_data.shape)
    # user_data.dropna(axis = 0,how='any')
    # video_data.dropna(axis = 0,how='any')
    ### 处理数据
    for name in Dense_fea[:4]:
        user_data,newL = sepSparse(user_data,name)    #拆分并组成多列
        Dense_fea += newL   #对 multi 进行统一labelencoder编码
    Dense_fea = Dense_fea[4:]
    print("Dense特征为",Dense_fea)

    video_data["video_release_date"] = video_data["video_release_date"].apply(date_separate)
    for name in multi_fea:
        video_data[name] = video_data[name].apply(separate)    #拆分并组成list
        video_data=LabelEncode(video_data,name)   #对 multi 进行统一labelencoder编码,并统计导演、演员统计量
    Dense_fea += ['video_director_list0', 'video_actor_list0','video_second_class0']
    video_data[Dense_fea[:3]] = mms.fit_transform(video_data[Dense_fea[:3]])  #大数先归一化，防止被0

    train_data=pd.DataFrame()
    for history_data_path in glob.glob(history_behavior_root+'*'):
        history_data = pd.read_csv(history_data_path, delimiter='\t', low_memory=True)
        history_data.dropna(axis = 0,how='any')
        history_data['watch_start_time'] = history_data['watch_start_time'].apply(date_separate)
        # 合并数据
        history_data = pd.merge(history_data, user_data, on='user_id', how = "left")
        history_data = pd.merge(history_data, video_data, on='video_id', how = "left")
        history_data.dropna(axis = 0,how='any')    #缺失值不能填充为0，直接删除
        train_data=pd.concat((train_data,history_data),axis=0).reset_index(drop=True)
        print("完成读取:", history_data_path)
    
    # train_data=pd.read_csv('./traindata/history_data.csv', low_memory=True)
    # for feat in Sparse_fea:
    #     print(feat,train_data[feat].nunique())

    #处理test、userAction加上统计量特征
    del history_data
    test_data = pd.merge(test_data, video_data, on='video_id', how = "left")
    test_data = pd.merge(test_data, user_data, on='user_id', how = "left")
    names = ['user_id','video_id']
    newlabel=[f"{names[0][:5]}label{j}" for j in range(6)] + [f"{names[1][:5]}label{j}" for j in range(6*len(ACTION_LIST)-14)]
    print("train缺失值检查",train_data['country0'].sum())
    train_data,test_data=get_dense(train_data,test_data,names,6)    #添加特征函数
    print("train缺失值检查",train_data['country0'].sum())
    Dense_fea += newlabel
    USER_FEA = ['user_id','pt_d','is_share','watch_label']+Dense_fea+Sparse_fea+multi_fea+embed_fea

    # df_neg = train_data[train_data['is_watch'] == 0]
    # df_neg = df_neg.sample(frac=0.02, random_state=42, replace=False)    # 0.1是接近1：1
    # train_data = pd.concat((df_neg, train_data[train_data['is_watch'] == 1])).reset_index(drop=True)
    # print("检查负样本比例：",df_neg.shape[0]/train_data.shape[0],train_data.shape[0])

    train_data = train_data[train_data['is_watch'] == 1].reset_index(drop=True)    
    all=pd.concat((train_data,test_data),axis=0)[USER_FEA].reset_index(drop=True)
    del train_data
    all[Sparse_fea]=all[Sparse_fea].fillna(-1)
    all=all.fillna(0)
    all=fill(all)
    # all.fillna(method='ffill',inplace=True)

    print("all缺失值检查",all.isnull().sum().sum())
    all[Dense_fea] = mms.fit_transform(all[Dense_fea])
    for feat in Sparse_fea:
        all[feat] = lbe.fit_transform(all[feat].astype('str'))
        print(feat,all[feat].nunique())
    train_data,test_data=all.iloc[:-test_data.shape[0]],all.iloc[-test_data.shape[0]:]
    
    train_data.sample(frac=1, random_state=42).to_pickle("./traindata/train_data.pickle")  #read_pickle读取,3.8只能在3.8上读
    test_data.to_pickle("./traindata/test_data.pickle")
    train_data.sample(frac=1, random_state=42).to_csv("./traindata/train_data.csv",index=0)
    test_data.to_csv("./traindata/test_data.csv",index=0)
    print(train_data.shape, test_data.shape)
    print("耗时（min）：",(time.time()-start_time)/60)
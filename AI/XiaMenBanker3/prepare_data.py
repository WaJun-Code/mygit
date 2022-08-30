import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.decomposition import PCA,TruncatedSVD
import time,tqdm,math
import pandas as pd
import re
train_test_L = ["SSTJMZKF001","SSTJMZKF002","DECD21090102","DECD21090103","DECD21090104","DECD21090105","DECD21090106",
          "DECD21090107","DECD21090108","90318011","91318017","HD1D001"]
#, 'prod1_14','prod1_15'
prod_Fea = ['prod1_0', 'prod1_1', 'prod1_2','prod1_3', 'prod1_5','prod1_4', 'prod1_9', 'prod1_10', 'prod1_11', 'prod1_16','prod2_3', 'prod2_4', 'prod2_5','prod2_6', 'prod2_7', 'prod2_8', 'prod2_9']
embed_dim = 16

start_time=time.time()
x_train,y_train = pd.read_csv("./Data_A/Data_main/x_train.csv"),pd.read_csv("./Data_A/Data_main/y_train.csv")
xy_test_A = pd.read_csv("./Data_B/Data_main/y_test_A.csv")
x_test = pd.read_csv("./Data_B/Data_main/x_test_B.csv")
x_train['a3'] = x_train['a3'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)))   #日期数据的处理
x_test['c3'] = x_test['c3'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)))
xy_test_A['c3'] = xy_test_A['c3'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)))
x_train.columns = ['id','core_cust_id','prod_code','a2','date']
x_test.columns = ['id','core_cust_id','prod_code','a2','date']
xy_test_A.columns = ['id','core_cust_id','prod_code','a2','date','y']
print("A榜test里有ABCD四类产品分别（条）:",(xy_test_A['a2']==1).sum(),(xy_test_A['a2']==2).sum(),(xy_test_A['a2']==3).sum(),(xy_test_A['a2']==4).sum())
print("A榜test里有ABCD四类产品分别（个）:",xy_test_A[xy_test_A['a2']==1]['prod_code'].nunique(),xy_test_A[xy_test_A['a2']==2]['prod_code'].nunique(),xy_test_A[xy_test_A['a2']==3]['prod_code'].nunique(),xy_test_A[xy_test_A['a2']==4]['prod_code'].nunique())
print("B榜test里有ABCD四类产品分别（条）:",(x_test['a2']==1).sum(),(x_test['a2']==2).sum(),(x_test['a2']==3).sum(),(x_test['a2']==4).sum())
print("B榜test里有ABCD四类产品分别（个）:",x_test[x_test['a2']==1]['prod_code'].nunique(),x_test[x_test['a2']==2]['prod_code'].nunique(),x_test[x_test['a2']==3]['prod_code'].nunique(),x_test[x_test['a2']==4]['prod_code'].nunique())
'''
L=x_train['prod_code'].unique().tolist()
print("test里有多少条数据产品id不在train表:",x_test['prod_code'].apply(lambda x :x not in L).sum())
print("test里有多少个产品id不在train表:",pd.DataFrame(x_test['prod_code'].unique())[0].apply(lambda x :x not in L).sum() ,x_test['prod_code'].nunique())
L=x_train['core_cust_id'].unique().tolist()
print("test里有多少条数据客户id不在train表:",x_test['core_cust_id'].apply(lambda x :x not in L).sum())
print("test里有多少个客户id不在train表:",pd.DataFrame(x_test['core_cust_id'].unique())[0].apply(lambda x :x not in L).sum() ,x_test['core_cust_id'].nunique())
'''
xy_train = pd.merge(x_train,y_train,how='left',on='id')
data_df = pd.concat((xy_train,xy_test_A,x_test)).fillna(-1).reset_index(drop=True)   #,x_test
data_other={}
for f in ['d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']:
    data_other[f] = pd.read_csv("./Data_A/Data_other/{}.csv".format(f),low_memory=False)
for f in ['d_bc','e_bc','f_bc','n_bc','o_bc','p_bc','q_bc','r_bc','s_bc']:
    data_other[f] = pd.read_csv("./Data_B/Data_other/{}.csv".format(f),low_memory=False)
for f in ['d','e','f','n','o','p','q','r','s']:
    data_other[f] = pd.concat((data_other[f],data_other[f+'_bc'])).reset_index(drop=True)
    del data_other[f+'_bc']

#理财产品信息表的预处理
data_other['g'].columns=['prod_code']+[f'prod1_{i}' for i in [0,1,2,3,4,5,7,15,16]]
data_other['g']['class']=1
data_other['h'].columns=['prod_code']+[f'prod1_{i}' for i in [0,1,2,3,5,7,8,16]]
data_other['h']['class']=2
data_other['i'].columns=['prod_code']+[f'prod1_{i}' for i in [0,1,2,3,5,6,7,15,16]]
data_other['i']['class']=3
data_other['j'].columns=['prod_code']+[f'prod1_{i}' for i in [0,1,2,3,5,9,10,11,12,13,14,15,16]]
data_other['j']['class']=4
prod1_df = pd.concat((data_other['j'],data_other['i'],data_other['h'],data_other['g'])).reset_index(drop=True)
data_other['k'].columns=['prod_code']+[f'prod2_{i}' for i in range(11)]
data_other['k']['class']=1
data_other['l'].columns=['prod_code']+[f'prod2_{i}' for i in [0,1,2,3,4,5,10]]
data_other['l']['class']=2
data_other['m'].columns=['prod_code']+[f'prod2_{i}' for i in [0,1,2,3,4,5,6,9,10]]
data_other['m']['class']=3
prod2_df = pd.concat((data_other['k'],data_other['l'],data_other['m'])).reset_index(drop=True)
tmp = prod1_df.merge(prod2_df.loc[prod2_df['prod2_6'].dropna().index],how='left',on='prod_code')
for i in range(prod1_df.shape[0]):
    if pd.isna(prod1_df.loc[i,'prod1_14']):
        if prod1_df.loc[i,'prod1_10']!=0.001100:prod1_df.loc[i,'prod1_14']=50*(prod1_df.loc[i,'prod1_10']+prod1_df.loc[i,'prod1_11'])
        if pd.isna(tmp.loc[i,'prod2_6'])==False:prod1_df.loc[i,'prod1_14']=100*tmp.loc[i,'prod2_6']
prod1_df.to_csv("./Data_A/prod1_df.csv",index=False)
prod2_df.to_csv("./Data_A/prod2_df.csv",index=False)
#理财产品交易信息表预处理
n,o,p,q = data_other['n'].copy(),data_other['o'].copy(),data_other['p'].copy(),data_other['q'].copy()
n.columns=[f'nopq_{i}' for i in [0,1,2]]+['core_cust_id','prod_code']+[f'nopq_{i}' for i in [3,4,7,6,8,11]]
n['class']=1
o.columns=[f'nopq_{i}' for i in [0,1,2]]+['core_cust_id','prod_code']+[f'nopq_{i}' for i in [3,4,6,7,8,9,11]]
o['class']=2
p.columns=[f'nopq_{i}' for i in [0,1,2]]+['core_cust_id','prod_code']+[f'nopq_{i}' for i in [3,4,5,6,7,10,11]]
p['class']=4
q.columns=[f'nopq_{i}' for i in [0,1,2]]+['core_cust_id','prod_code']+[f'nopq_{i}' for i in [3,4,7,6,11]]
q['class']=3
nopq_df = pd.concat((p,o,n,q))
del n,o,p,q
nopq_df.to_csv("./Data_A/nopq_df.csv",index=False)

prod1_df = pd.read_csv("./Data_A/prod1_df.csv",low_memory=False)
prod2_df = pd.read_csv("./Data_A/prod2_df.csv",low_memory=False)
#d表的统计与合并
for f in [data_other['d']]:
    data_df = pd.merge(data_df, f,how='left',on='core_cust_id')
print("完成d表merge",data_df.shape)
#f表的统计与合并,只留下年月日均和时点余额
data_other['f'] = data_other['f'].loc[data_other['f'].iloc[:,2:-1].dropna(how='all').index]     #去掉资产信息全为nan的
data_other['f']['f1'] = data_other['f']['f1'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)))
for f in [f'f{i}' for i in range(2,22)]:data_other['f'][f] = data_other['f'][f].apply(lambda x:float(re.sub(',','',x)) if pd.isna(x)==0 else x)   #全部进行数字金额转换float

data_other['f'][[f'f{i+2}' for i in range(5)]] = data_other['f'][[f'f{i+7}' for i in range(5)]] #只留下余额即可
data_other['f'].drop(columns=[f'f{i}' for i in np.linspace(7,21,15).astype(int)],inplace=True)
data_other['f']['fs'] = data_other['f'][[f'f{i}' for i in np.linspace(2,6,5).astype(int)]].sum(axis=1)   #统计五种的和
data_other['f']['f22'] = data_other['f']['f22'].apply(lambda x:x+71)
data_other['f'].rename(columns={'f22':'date'},inplace=True)
tmp = data_other['f'].drop(columns=['f1']).copy()
tmp['date'] = tmp['date'].apply(lambda x:x+100)
tmp.columns = ['core_cust_id']+[f'f_{i+2}' for i in range(5)]+['date','fs_']
data_other['f'] = pd.merge(data_other['f'],tmp,how='left',on=['core_cust_id','date'])
data_other['f'][[f'f_{i+2}' for i in range(5)]+['fs_']] = data_other['f'][[f'f{i+2}' for i in range(5)]+['fs']].values-data_other['f'][[f'f_{i+2}' for i in range(5)]+['fs_']].values   #统计上下两个月的余额之差
print(data_other['f'].columns)
data_df = pd.merge(data_df,data_other['f'],how='left',on=['core_cust_id','date'])
print("完成f表merge",data_df.shape)

#理财产品交易信息表预处理
data_other['n']['n7'] = data_other['n']['n7'].apply(lambda x: str(x).replace(',','')).astype('float')
data_other['o']['o7'] = data_other['o']['o7'].apply(lambda x: str(x).replace(',','')).astype('float')
data_other['p']['p7'] = data_other['p']['p7'].apply(lambda x: str(x).replace(',','')).astype('float')
data_other['q']['q7'] = data_other['q']['q7'].apply(lambda x: str(x).replace(',','')).astype('float')

data_other['r']['r5'] = data_other['r']['r5'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)[:8]))
data_other['s']['s4'] = data_other['s']['s4'].apply(lambda x:float(re.sub(',','',x)))
data_other['s']['s7'] = data_other['s']['s7'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)))
#产品交易统一表，后面做embedding的时候用
nopq_df = pd.read_csv("./Data_A/nopq_df.csv",low_memory=False)
nopq_df['nopq_4'] = nopq_df['nopq_4'].apply(lambda x:float(re.sub(',','',x)))   #处理金额信息
nopq_df['nopq_8'] = nopq_df['nopq_8'].apply(lambda x:float(re.sub(',','',x)) if pd.isna(x)==0 else x)
nopq_df['nopq_9'] = nopq_df['nopq_9'].apply(lambda x:float(re.sub(',','',x)) if pd.isna(x)==0 else x)
nopq_df['date'] = nopq_df['nopq_11']
nopq_df = nopq_df.drop(columns=['nopq_11'])   #将时间信息放到最后一列
# nopq_df = nopq_df[nopq_df['prod_code'].apply(lambda x :x in L)]
#将s表的s3和s6处理为客户id，这样处理效果并不好
# s3 = data_other['s'].copy()
# s3['s4'] = -1*s3['s4']
# s3.columns = ['s1','s2','core_cust_id','s4','s5','s6','s7']
# s3 = s3.drop(columns=['s6'])
# data_other['s'].columns = ['s1','s2','s3','s4','s5','core_cust_id','s7']
# data_other['s'] = data_other['s'].drop(columns=['s3'])
# data_other['s'] = pd.concat((s3,data_other['s'])).dropna(how='any')[['core_cust_id','s2','s4','s7']]
# 对s表的处理不同，需要group后count操作，但也要记得统一date键值
def custprod_group(kstr,dstr,listgroup,keystr,actlist):   #一次只能有一个keystr，做之前所有时间的统计特征
    df = data_other[kstr]
    _0701 = df[df[dstr]<20210701].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0701['date'] = 20210701
    _0801 = df[df[dstr]<20210801].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0801['date'] = 20210801
    _0901 = df[df[dstr]<20210901].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0901['date'] = 20210901
    _1001 = df[df[dstr]<20211001].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _1001['date'] = 20211001
    _1201 = df[df[dstr]<20211201].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _1201['date'] = 20211201
    _df = pd.concat((_0701,_0801,_0901,_1001,_1201)).reset_index(drop=True)
    _df.columns = listgroup+[kstr+'_' + keystr+'_'+i for i in actlist]+['date']
    return _df
def custprod_1group(kstr,dstr,listgroup,keystr,actlist):   #一次只能有一个keystr，做之前一个月时间的统计特征
    df = data_other[kstr]
    _0701 = df[df[dstr].apply(lambda x:20210600<x<20210701)].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0701['date'] = 20210701
    _0801 = df[df[dstr].apply(lambda x:20210700<x<20210801)].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0801['date'] = 20210801
    _0901 = df[df[dstr].apply(lambda x:20210800<x<20210901)].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _0901['date'] = 20210901
    _1001 = df[df[dstr].apply(lambda x:20210900<x<20211001)].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _1001['date'] = 20211001
    _1201 = df[df[dstr].apply(lambda x:20211100<x<20211201)].groupby(listgroup)[keystr].agg(actlist).reset_index()
    _1201['date'] = 20211201
    _df = pd.concat((_0701,_0801,_0901,_1001,_1201)).reset_index(drop=True)
    _df.columns = listgroup+[kstr+'_' + keystr+'_1m_'+i for i in actlist]+['date']
    return _df
#数据全部处理完毕，d、e、f、s表通过客户id来分别merge，prod1、prod2通过产品id，nopq、r通过【客户id，产品id】merge
def get_now(df):   #获取离train数据时间最近的信息
    colname = df.columns.tolist()   #要求date在最后一列
    e_0701 = df[df[colname[-1]]<20210701].sort_values(colname[-1]).drop_duplicates(['core_cust_id'],keep='last')   #选取训练集对应最近信息
    e_0701[colname[-1]]=20210701
    e_0801 = df[df[colname[-1]]<20210801].sort_values(colname[-1]).drop_duplicates(['core_cust_id'],keep='last')
    e_0801[colname[-1]]=20210801
    e_0901 = df[df[colname[-1]]<20210901].sort_values(colname[-1]).drop_duplicates(['core_cust_id'],keep='last')
    e_0901[colname[-1]]=20210901
    e_1001 = df[df[colname[-1]]<20211001].sort_values(colname[-1]).drop_duplicates(['core_cust_id'],keep='last')
    e_1001[colname[-1]]=20211001
    e_1201 = df[df[colname[-1]]<20211201].sort_values(colname[-1]).drop_duplicates(['core_cust_id'],keep='last')
    e_1201[colname[-1]]=20211201
    df = pd.concat((e_0701,e_0801,e_0901,e_1001)).reset_index(drop=True)
    df.columns = colname[:-1]+['date']         #注意改列名为date
    return df

data_1s3 = custprod_1group('s','s7',['s3'],'s4',['sum','count','mean'])
data_1s3.rename(columns={'s3':'core_cust_id'}, inplace=True)
data_3s2 = custprod_1group('s','s7',['s3'],'s2',['mean'])   #对借贷交易类型代码求mean，'s_s2_1m_mean','s_s2_6_1m_mean'
data_3s2.rename(columns={'s3':'core_cust_id'}, inplace=True)
data_other['s']['s4_6'] = data_other['s']['s4']   #引入一个新列，防止聚合重名
data_1s6 = custprod_1group('s','s7',['s6'],'s4_6',['sum','count','mean'])
data_1s6.rename(columns={'s6':'core_cust_id'}, inplace=True)
data_other['s']['s2_6'] = data_other['s']['s2']   #引入一个新列，防止聚合重名
data_6s2 = custprod_1group('s','s7',['s6'],'s2_6',['mean'])
data_6s2.rename(columns={'s6':'core_cust_id'}, inplace=True)
#s表的统计与合并
for f in [data_1s3,data_1s6,data_3s2,data_6s2]:   #还未考虑 s2 类别 list 如何利用
    data_df = pd.merge(data_df, f,how='left',on=['core_cust_id','date'])
print("完成s表merge",data_df.shape)
df = pd.concat((xy_train,xy_test_A)).reset_index(drop=True)
# 正好统计一下label的统计信息
data_other['y'] = df
print(df.shape)
#data_other['y']['date'] = data_other['y']['date'].apply(lambda x:x+1)
user_y = custprod_group('y','date',['core_cust_id'],'y',['count','mean','sum'])
user_y.columns = ['core_cust_id','u_y_count','u_y_mean','u_y_sum','date']
data_df = pd.merge(data_df,user_y,how='left',on=['core_cust_id','date'])
item_y = custprod_group('y','date',['prod_code'],'y',['count','mean','sum'])
item_y.columns = ['prod_code','p_y_count','p_y_mean','p_y_sum','date']
data_df = pd.merge(data_df,item_y,how='left',on=['prod_code','date'])
print("完成label统计与merge",data_df.shape)

data_df = data_df.fillna(0)   #金额和label统计可以先全部fill为0，因为NaN的可以认为是不活跃用户钱少

# data_e = custprod_group('e','e2',['core_cust_id'],'e1',['mean','count'])   #说用之前的数据效果不佳，只用一个月的即可
data_1e = custprod_1group('e','e2',['core_cust_id'],'e1',['mean'])
data_df = pd.merge(data_df, data_1e,how='left',on=['core_cust_id','date'])
# data_other['e'] = get_now(data_other['e'])
print("完成e表merge",data_df.shape)
#理财产品信息表的预处理
dataL_prod = data_df['prod_code'].unique().tolist()   #用产品信息筛一遍
for f in [prod1_df.drop(columns=['class']),prod2_df.drop(columns=['class'])]:   #,data_other['g'],data_other['j'],data_other['k'] 
    f = f[f['prod_code'].apply(lambda x:x in dataL_prod)]
    tmp = f.describe().T
    useful_cols = list(tmp[(tmp['std'] != 0) & (pd.isna(tmp['std']) == 0)].index)   #找到在train里的prod对应数据的方差不为0和nan的列
    useful_cols = [c for c in useful_cols if c not in ['g9', 'h8', 'i9', 'j13', 'k11', 'l7','m9','date','class']]  #删掉产品数据日期
    print(useful_cols)
    if len(useful_cols) > 0:data_df = data_df.merge(f[['prod_code']+useful_cols], how='left', on='prod_code')
print("完成ghijklm表merge",data_df.shape)
#APP点击信息表：
data_r = custprod_group('r','r5',['core_cust_id'],'prod_code',['nunique','count'])
data_df = pd.merge(data_df,data_r,how='left',on=['core_cust_id','date'])
# data_r = custprod_group('r','r5',['prod_code'],'core_cust_id',['nunique','count'])
# data_df = pd.merge(data_df,data_r,how='left',on=['prod_code','date'])
# data_other['r'] = data_other['r'].drop_duplicates(['core_cust_id','prod_code'])[['core_cust_id','prod_code','r3']]   #然后录入r3类别特征
# data_df = pd.merge(data_df,data_other['r'],how='left',on=['core_cust_id','prod_code'])
print("完成r表merge",data_df.shape)
#产品交易表

dateL = {'n':'n11','o':'o12','p':'p12','q':'q10'}
for f in ['n','o','p','q' ]: #为什么交易流水表用了当天信息也会穿越
    nopq_prod = custprod_group( f, dateL[f],['core_cust_id'],'prod_code',['count','nunique'])
    data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','date'])
    nopq_prod = custprod_group( f, dateL[f],['core_cust_id'], f+'7',['mean','sum'])
    data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','date'])
    nopq_prod = custprod_1group( f, dateL[f],['core_cust_id'],'prod_code',['count','nunique'])
    data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','date'])
    nopq_prod = custprod_1group( f, dateL[f],['core_cust_id'], f+'7',['mean','sum'])
    data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','date'])
data_other['nopq'] = nopq_df
nopq_prod = custprod_1group( 'nopq', 'date',['prod_code'], 'nopq_3',['mean'])
nopq_prod.columns = ['prod_code','nopq_3_mean','date']
data_df = pd.merge(data_df,nopq_prod,how='left',on=['prod_code','date'])
nopq_prod = custprod_1group( 'nopq', 'date',['prod_code'], 'nopq_5',['mean'])
nopq_prod.columns = ['prod_code','nopq_5_mean','date']
data_df = pd.merge(data_df,nopq_prod,how='left',on=['prod_code','date'])
for f in ['nopq_3','nopq_5']:
    data_y = custprod_group('nopq','date',['core_cust_id'],f,['min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['core_cust_id','date'])
    #做差
    data_df['nopq_{}_min'.format(f)] = data_df['{}_mean'.format(f)]>=data_df['nopq_{}_min'.format(f)]
    data_df['nopq_{}_max'.format(f)] = data_df['{}_mean'.format(f)]<=data_df['nopq_{}_max'.format(f)]
    data_df['{}_mean'.format(f)] = data_df['nopq_{}_min'.format(f)] & data_df['nopq_{}_max'.format(f)]
    data_df = data_df.drop(columns=['nopq_{}_min'.format(f),'nopq_{}_max'.format(f)])
nopq_prod = custprod_1group('nopq', 'date',['core_cust_id','prod_code'],'nopq_6',['min','max'])
data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','prod_code','date'])
nopq_prod = custprod_1group('nopq', 'date',['core_cust_id','prod_code'],'nopq_7',['min','max'])
data_df = pd.merge(data_df,nopq_prod,how='left',on=['core_cust_id','prod_code','date'])

#对各个产品端进行细致统计
# nopq_prod = custprod_1group('nopq', 'date',['prod_code'],'core_cust_id',['count','nunique'])
# nopq_4 = custprod_1group('nopq', 'date',['prod_code'], 'nopq_4',['mean','sum'])   #同理于s，用同一函数做统计特征
# for f in [nopq_prod,nopq_4 ]:    #nopq表开始merge
#     data_df = pd.merge(data_df,f,how='left',on=['prod_code','date'])
# for f in ['mean','max','min']:   #客户-产品做差
#     data_df['nopq_nopq_4_1m_{}_y'.format(f)] = data_df['nopq_nopq_4_{}'.format(f)]-data_df['nopq_nopq_4_1m_{}_y'.format(f)]
# for key in ['nopq_3','nopq_5','nopq_8','nopq_9']:
#     nopq_4 = custprod_group('nopq', 'date',['prod_code'], key,['mean'])
#     data_df = pd.merge(data_df,nopq_4,how='left',on=['prod_code','date'])

print("完成nopq表merge",data_df.shape)
print("耗时(min):",(time.time()-start_time)/60)
#粘上emb和相似度特征
def item_cf(df,user_col,item_col):  # 录入 df 为user-item历史行为序列，此处用评分5的
    user_item = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item[user_col],user_item[item_col]))   #{core_cust_id : item_list}

    sim_item = {}    #用来存 item_cf 协同过滤对应 相似度
    item_cnt = defaultdict(int)   #用来存每个item共被多少个user过
    for user,items in tqdm.tqdm(user_item_dict.items()):
        for item in items:
            item_cnt[item] += 1
            sim_item.setdefault(item,{})
            for relate_item in items:
                if item == relate_item:   #被同一user访问过的item之间有一定关联，故后面+1
                    continue
                sim_item[item].setdefault(relate_item,0)
                sim_item[item][relate_item] += 1/math.log(1+len(items))   #引入user看过item的总数，减小爱好广泛用户的影响

    sim_item_corr = sim_item.copy()
    for i,related_items in tqdm.tqdm(sim_item.items()):
        for j,cij in related_items.items():
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])

    return sim_item_corr, user_item_dict     #里面只存了有过交互的item-item对，无法解决item冷启动问题
def user2vec(df,glist):
    df = df.fillna(0)
    df0 = df.groupby(glist)['core_cust_id'].agg(list).reset_index()
    df0.columns = glist+['vec_core_cust_ids']
    w2v = Word2Vec(df0['vec_core_cust_ids'].values.tolist(), size=embed_dim, sg=1, window=10000, seed=42, workers=48, min_count=0, iter=20)
    user_vec, item_vec = {}, {}
    for i in df['core_cust_id'].unique():user_vec[i] = w2v[i].tolist()
    print("user字典检查：",len(user_vec)==df['core_cust_id'].nunique())
    for i in df0.values:
        item_vec[i[0]] = np.zeros([embed_dim,])
        for j in i[-1]:item_vec[i[0]] += np.array(user_vec[j])/len(i[-1])
        item_vec[i[0]] = item_vec[i[0]].tolist()
    print("item字典检查：",len(item_vec)==df['prod_code'].nunique())
    return user_vec, item_vec
def SVD_emb(Emb,name):
    indexs=Emb.index.tolist()
    clf=TruncatedSVD(4)
    Emb=clf.fit_transform(Emb)
    Emb=pd.DataFrame(data=Emb,columns=[f"{name}{i}" for i in range(4)],index=indexs)
    return Emb
def merge_sim(df,item_sim_dict, user_item):   #将itemcf得到的sim矩阵按user历史mask后merge进特征里
    df_sim = pd.DataFrame(np.zeros([df.shape[0],len(item_sim_dict)]),columns=list(item_sim_dict.keys()))
    i = 0
    for x in tqdm.tqdm(df[['core_cust_id','prod_code']].values.tolist() ):
        if x[1] in list(item_sim_dict.keys()):
            k_1=list(item_sim_dict[x[1]].keys())
            if x[0] in list(user_item.keys()):
                item_l = user_item[x[0]]
                item_l = pd.DataFrame(item_l)[0].unique().tolist()
                for j in user_item[x[0]]:item_l+=list(item_sim_dict[j].keys())   #历史看过的，以及历史看过类似的list
                item_l = pd.DataFrame(item_l)[0].unique().tolist()
                for j in item_l:
                    if j==x[1]:df_sim.loc[i,x[1]] += 1
                    elif j in k_1 and item_sim_dict[x[1]][j]>0.08:df_sim.loc[i,j] += item_sim_dict[x[1]][j]
        i+=1
    df_sim = SVD_emb(df_sim,'CFsim')
    df_sim.columns = [f"CFsim{i}" for i in range(df_sim.shape[1])]
    df = pd.concat((df,df_sim),axis=1)
    return df


def merge_emb(df,data_df,d):
    data_df = data_df[data_df["date"]==d][["core_cust_id","prod_code","date"]]
    df = df[df["date"]<d][["core_cust_id","prod_code","date"]]
    if df.shape[0]==0:
        data_df[[f"user_emb{i}" for i in range(embed_dim)]] = 0
        #data_df[["cos","dist"]]=-999   #这两个不能填0
        return data_df
    user_vec, item_vec = user2vec(df,['prod_code'])
    #补齐user2vec
    item_L, user_L=data_df['prod_code'].unique().tolist(), data_df['core_cust_id'].unique().tolist()
    for i in list(user_vec.keys()):
        if i not in user_L:del user_vec[i]   #删除多余的
    for i in list(item_vec.keys()):
        if i not in item_L:del item_vec[i]
    k=0
    for i in item_L:
        if i not in item_vec.keys():
            k+=1
            item_vec[i] = np.zeros([embed_dim,]).tolist()    #补充缺失的id，1是因为后面点积不影响另外一个vec
    print("item_vec缺失num：item", k, len(item_vec))
    k=0
    for i in user_L:
        if i not in user_vec.keys():
            k+=1
            user_vec[i] = np.zeros([embed_dim,]).tolist()    #补充缺失的id
    print("user_vec缺失num：user", k, len(user_vec))
    print("user总字典检查：",len(user_vec)==len(user_L))
    print("item总字典检查：",len(item_vec)==len(item_L))
    #粘上做了点积后的emb
    df_user_emb_y = pd.DataFrame(list(user_vec.values()),index =user_vec.keys()).reset_index().rename(columns={"index":"core_cust_id"})
    df_item_emb_y = pd.DataFrame(list(item_vec.values()),index =item_vec.keys()).reset_index().rename(columns={"index":"prod_code"})
    data0 = pd.merge(data_df,df_user_emb_y,how='left',on="core_cust_id").drop(columns=["core_cust_id","prod_code","date"]).values
    data1 = pd.merge(data_df,df_item_emb_y,how='left',on="prod_code").drop(columns=["core_cust_id","prod_code","date"]).values
    #norm = np.linalg.norm(data0,axis=1)*np.linalg.norm(data1,axis=1)+1e-9
    #data_df["cos"] = np.sum(data0*data1,axis=1)/norm
    #norm = np.linalg.norm(data0-data1,axis=1)
    #data_df["dist"] = 1/(1+5*norm)
    #data_df[["cos","dist"]] = data_df[["cos","dist"]].fillna(-999)
    data_df[[f"user_emb{i}" for i in range(embed_dim)]] = data0   #粘上user的emb
    #data_df[[f"u_p_emb{i}" for i in range(embed_dim)]] = data0*data1   #粘上交叉点积的emb
    del data0,data1
    return data_df
def get_sim(df,data_df,d):
    data_df = data_df[data_df["date"]==d][["core_cust_id","prod_code","date"]]
    df = df[df["date"]<d][["core_cust_id","prod_code","date"]]
    if df.shape[0]==0:
        data_df[[f"CFsim{i}" for i in range(4)]] = 0
        return data_df
    item_sim_dict, user_item = item_cf(df, 'core_cust_id','prod_code')
    # item_sim = pd.DataFrame(np.zeros([df['prod_code'].nunique(),len(item_sim_dict)]),columns=list(item_sim_dict.keys()),index=df['prod_code'].unique().tolist())
    # for k0,vs in tqdm.tqdm(item_sim_dict.items()):
    #     item_sim.loc[k0,k0]=1
    #     for k1,v in vs.items():item_sim.loc[k0,k1] = v
    data_df = merge_sim(data_df,item_sim_dict,user_item)
    return data_df
df = df[df['y']==1]
emb_0701 = merge_emb(df,data_df,20210701)
emb_0801 = merge_emb(df,data_df,20210801)
emb_0901 = merge_emb(df,data_df,20210901)
emb_1001 = merge_emb(df,data_df,20211001)
emb_1201 = merge_emb(df,data_df,20211201)
emb = pd.concat((emb_0701,emb_0801,emb_0901,emb_1001,emb_1201))
data_df = pd.merge(data_df,emb, how = 'left',on=["core_cust_id","prod_code","date"])
print("完成user2vec表merge",data_df.shape)
#sim_0701 = get_sim(df,data_df,20210701)
#sim_0801 = get_sim(df,data_df,20210801)
#sim_0901 = get_sim(df,data_df,20210901)
#sim_1001 = get_sim(df,data_df,20211001)
#sim_1201 = get_sim(df,data_df,20211201)
#sim = pd.concat((sim_0701,sim_0801,sim_0901,sim_1001,sim_1201))
#data_df = pd.merge(data_df,sim, how = 'left',on=["core_cust_id","prod_code","date"])
#print("完成itemcf表merge",data_df.shape)
print("数据处理耗时(min):",(time.time()-start_time)/60)

#以下缺失值处理不能直接填0
df = data_df.iloc[:-x_test.shape[0]].copy()   #得copy，不然date会随之变化，引发大问题
#df = data_df.iloc[:xy_train.shape[0]].copy()
print(df.shape)
data_other['y'] = df[df['y']==1]
print(data_other['y'].shape)
data_y = custprod_group('y','date',['core_cust_id'],'prod_code',['nunique'])
data_df = pd.merge(data_df,data_y,how='left',on=['core_cust_id','date'])
for f in ['p_y_count','p_y_mean','p_y_sum','prod1_14','prod1_15','a2']:
    data_y = custprod_group('y','date',['core_cust_id'],f,['mean','min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['core_cust_id','date'])
    #做差
    data_df['y_{}_mean'.format(f)] = np.abs(data_df['y_{}_mean'.format(f)]-data_df[f])
    data_df['y_{}_min'.format(f)] = data_df['y_{}_min'.format(f)]<=data_df[f]
    data_df['y_{}_max'.format(f)] = data_df['y_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['y_{}_max'.format(f)] & data_df['y_{}_min'.format(f)]).to_frame()
    df0.columns = ['y_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['y_{}_min'.format(f),'y_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)
data_y = custprod_1group('y','date',['prod_code'],'core_cust_id',['nunique'])
data_df = pd.merge(data_df,data_y,how='left',on=['prod_code','date'])
data_y = custprod_group('y','date',['prod_code'],'core_cust_id',['nunique'])
data_df = pd.merge(data_df,data_y,how='left',on=['prod_code','date'])
for f in ['u_y_count','u_y_mean','u_y_sum','d1','d2','d3', 'e_e1_1m_mean','s_s2_1m_mean','s_s2_6_1m_mean','fs','fs_', 's_s4_1m_mean', 's_s4_1m_sum', 's_s4_1m_count', 's_s4_6_1m_mean', 's_s4_6_1m_sum', 's_s4_6_1m_count']:
    data_y = custprod_group('y','date',['prod_code'],f,['mean','min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['prod_code','date'])
    data_df['y_{}_mean'.format(f)] = np.abs(data_df['y_{}_mean'.format(f)]-data_df[f])
    data_df['y_{}_min'.format(f)] = data_df['y_{}_min'.format(f)]<=data_df[f]
    data_df['y_{}_max'.format(f)] = data_df['y_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['y_{}_max'.format(f)] & data_df['y_{}_min'.format(f)]).to_frame()
    df0.columns = ['y_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['y_{}_min'.format(f),'y_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)
print("完成好评差评客户产品分别信息统计:",data_df.shape)   #这个特征居然没有下面两个有用，下面两个在ctb里提了13k

df = nopq_df
df['date'] = df['date'].apply(lambda x:(x>20210701)*(x-20210601)//100*100+20210701 ) #通过该公式直接转换成一致的date格式
df = pd.merge(df,data_other['f'],how='left',on=['core_cust_id','date'])
for f in [data_1s3,data_1s6,data_3s2,data_6s2]:df = pd.merge(df, f,how='left',on=['core_cust_id','date'])  #'s_s4_1m_mean', 's_s4_1m_sum', 's_s4_1m_count', 's_s4_6_1m_mean', 's_s4_6_1m_sum', 's_s4_6_1m_count', 'e_e1_1m_mean'
df = pd.merge(df,item_y,how='left',on=['prod_code','date'])  #'p_y_count','p_y_mean','p_y_sum',
df = pd.merge(df,user_y,how='left',on=['core_cust_id','date'])  #'u_y_count','u_y_mean','u_y_sum',
df = df.fillna(0)   #冷启动的prod对应特征均为0即可
prod1_df.rename(columns={'class':'a2'},inplace=True)
df = pd.merge(df,prod1_df,how='left',on='prod_code')
df = pd.merge(df,data_other['d'],how='left',on='core_cust_id')
df = pd.merge(df,data_1e,how='left',on=['core_cust_id','date'])
data_other['nopq'] = df
for f in ['p_y_count','p_y_mean','p_y_sum','prod1_14','prod1_15','a2']:   #merge后与相应产品信息做差
    data_y = custprod_group('nopq','date',['core_cust_id'],f,['mean','min','max'])   #由于前面用了date格式转换，此处也可不用函数，直接对date聚合，表示一个月的统计信息
    data_df = pd.merge(data_df,data_y,how='left',on=['core_cust_id','date'])
    data_df['nopq_{}_mean'.format(f)] = np.abs(data_df['nopq_{}_mean'.format(f)]-data_df[f])
    data_df['nopq_{}_min'.format(f)] = data_df['nopq_{}_min'.format(f)]<=data_df[f]
    data_df['nopq_{}_max'.format(f)] = data_df['nopq_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['nopq_{}_max'.format(f)] & data_df['nopq_{}_min'.format(f)]).to_frame()
    df0.columns = ['nopq_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['nopq_{}_min'.format(f),'nopq_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)
for f in ['u_y_count','u_y_mean','u_y_sum','d1','d2','d3', 'e_e1_1m_mean','s_s2_1m_mean','s_s2_6_1m_mean','fs','fs_', 's_s4_1m_mean', 's_s4_1m_sum', 's_s4_1m_count', 's_s4_6_1m_mean', 's_s4_6_1m_sum', 's_s4_6_1m_count']:
    data_y = custprod_group('nopq','date',['prod_code'],f,['mean','min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['prod_code','date'])
    data_df['nopq_{}_mean'.format(f)] = np.abs(data_df['nopq_{}_mean'.format(f)]-data_df[f])
    data_df['nopq_{}_min'.format(f)] = data_df['nopq_{}_min'.format(f)]<=data_df[f]
    data_df['nopq_{}_max'.format(f)] = data_df['nopq_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['nopq_{}_max'.format(f)] & data_df['nopq_{}_min'.format(f)]).to_frame()
    df0.columns = ['nopq_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['nopq_{}_min'.format(f),'nopq_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)
print("完成nopq客户产品分别信息统计:",data_df.shape)

df = data_other['r']
df.rename(columns={'r5':'date'},inplace=True)
df['date'] = df['date'].apply(lambda x:(x>20210701)*(x-20210601)//100*100+20210701 ) #通过该公式直接转换成一致的date格式
df = pd.merge(df,data_other['f'],how='left',on=['core_cust_id','date'])
for f in [data_1s3,data_1s6,data_3s2,data_6s2]:df = pd.merge(df, f,how='left',on=['core_cust_id','date'])  #如果这一系列mean不能提分，则加入std，用3thigma原则判定true、false
df = pd.merge(df,item_y,how='left',on=['prod_code','date'])
df = pd.merge(df,user_y,how='left',on=['core_cust_id','date'])
df = df.fillna(0)
df = pd.merge(df,prod1_df,how='left',on='prod_code')
df = pd.merge(df,data_other['d'],how='left',on='core_cust_id')
df = pd.merge(df,data_1e,how='left',on=['core_cust_id','date'])
data_other['r'] = df
for f in ['p_y_count','p_y_mean','p_y_sum','prod1_14','prod1_15','a2']:   #merge后与相应产品信息做差
    data_y = custprod_group('r','date',['core_cust_id'],f,['mean','min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['core_cust_id','date'])
    data_df['r_{}_mean'.format(f)] = np.abs(data_df['r_{}_mean'.format(f)]-data_df[f])
    data_df['r_{}_min'.format(f)] = data_df['r_{}_min'.format(f)]<=data_df[f]
    data_df['r_{}_max'.format(f)] = data_df['r_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['r_{}_max'.format(f)] & data_df['r_{}_min'.format(f)]).to_frame()
    df0.columns = ['r_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['r_{}_min'.format(f),'r_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)
for f in ['u_y_count','u_y_mean','u_y_sum','d1','d2','d3', 'e_e1_1m_mean','s_s2_1m_mean','s_s2_6_1m_mean','fs','fs_', 's_s4_1m_mean', 's_s4_1m_sum', 's_s4_1m_count', 's_s4_6_1m_mean', 's_s4_6_1m_sum', 's_s4_6_1m_count']:
    data_y = custprod_group('r','date',['prod_code'],f,['mean','min','max'])
    data_df = pd.merge(data_df,data_y,how='left',on=['prod_code','date'])
    data_df['r_{}_mean'.format(f)] = np.abs(data_df['r_{}_mean'.format(f)]-data_df[f])
    data_df['r_{}_min'.format(f)] = data_df['r_{}_min'.format(f)]<=data_df[f]
    data_df['r_{}_max'.format(f)] = data_df['r_{}_max'.format(f)]>=data_df[f]
    df0 = (data_df['r_{}_max'.format(f)] & data_df['r_{}_min'.format(f)]).to_frame()
    df0.columns = ['r_{}_minmax'.format(f)]
    data_df = data_df.drop(columns=['r_{}_min'.format(f),'r_{}_max'.format(f)])
    data_df = pd.concat((data_df,df0),axis=1)

print("完成r客户产品分别信息统计:",data_df.shape)

del data_other['y'],data_other['nopq'],df
''''''
print("处理耗时(min):",(time.time()-start_time)/60,data_df.info())

data_df = data_df.drop(columns= prod_Fea)   #去掉产品统一信息
print("用方差drop前",data_df.shape)
train_Fea = data_df.drop(columns=['id','core_cust_id','prod_code','date']).columns.tolist()
drop_cols = [c for c in train_Fea if data_df[c].dtype != 'object' and data_df[c].std() == 0]   #去掉一些方差为0的f
data_df.drop(columns=drop_cols, inplace=True)
train_Fea = data_df.drop(columns=['id','core_cust_id','prod_code','date']).columns.tolist()
print("用方差drop后",data_df.shape)

xy_train,xy_test_A,x_test = data_df.iloc[:xy_train.shape[0]],data_df.iloc[xy_train.shape[0]:-x_test.shape[0]],data_df.iloc[-x_test.shape[0]:]
xy_train = xy_train.drop_duplicates(train_Fea)   #去重
print("缺失值检查：",xy_train.isnull().sum().sum(), xy_test_A.isnull().sum().sum(),x_test.isnull().sum().sum())
print(xy_train.shape, xy_test_A.shape, x_test.shape)
xy_train.to_csv("./Data_A/xy_train.csv",index=False)
xy_test_A.to_csv("./Data_A/xy_test_A.csv",index=False)
x_test.drop(columns=['y']).to_csv("./Data_A/x_test.csv",index=False)
print("耗时(min):",(time.time()-start_time)/60)

from utils import *
import numpy as np
import pandas as pd
import joblib,json
'''
train_xy.Date = pd.to_datetime(train_xy.Date)
train_xy['Date'] = train_xy['Date'].dt.strftime("%Y%m%d").astype(int)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import warnings,random,os,tqdm,re
import time,copy
warnings.filterwarnings("ignore")
train_fin = pd.read_csv("./train_files/financials.csv",low_memory=False)  #各个季度的季度情况
train_op = pd.read_csv("./train_files/options.csv",low_memory=False)
train_sec_pr = pd.read_csv("./train_files/secondary_stock_prices.csv",low_memory=False)#.rename(columns={'RowId':'DateCode'})
train_pr = pd.read_csv("./train_files/stock_prices.csv",low_memory=False)
train_trd = pd.read_csv("./train_files/trades.csv",low_memory=False).dropna(how='any').reset_index(drop=True)   #前一个交易周的市场总交易情况

stock_list = pd.read_csv("stock_list.csv",low_memory=False).rename(columns={'Section/Products':'Section','Close':'Close_MarketCapitalization'})
for f in ["17SectorName","17SectorCode","33SectorName","33SectorCode"]:
    stock_list[f] = stock_list[f].apply(lambda x:np.nan if x=='－' else x.strip())
valid_fin = pd.read_csv("./supplemental_files/financials.csv",low_memory=False)  #各个季度的季度情况
valid_op = pd.read_csv("./supplemental_files/options.csv",low_memory=False)
valid_sec_pr = pd.read_csv("./supplemental_files/secondary_stock_prices.csv",low_memory=False)
valid_pr = pd.read_csv("./supplemental_files/stock_prices.csv",low_memory=False)
valid_trd = pd.read_csv("./supplemental_files/trades.csv",low_memory=False).dropna(how='any').reset_index(drop=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Securities_Code_L = set(train_pr["SecuritiesCode"].unique().tolist() )

def reduce_mem(df):    # 节约内存的一个标配函数
    starttime=time.time()
    numerics=['int16','int32','int64','float16','float32','float64']
    start_mem=df.memory_usage().sum()/1024**2
    for col in df.columns:
        col_type=df[col].dtypes
        if col_type in numerics:
            c_min=df[col].min()
            c_max=df[col].max()
        else:continue
        if pd.isnull(c_min) or pd.isnull(c_max):continue
        if str(col_type)[:3]=='int':
            if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    #print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,100*(start_mem-end_mem)/start_mem, (time.time()-starttime)/60))
    return df
def seed_torch(seed=42):   #torch使结果可复现
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def process(train_fin,train_op,train_pr,train_sec_pr,train_trd ):
    for f in [train_fin,train_op,train_pr,train_sec_pr,train_trd ]:
        f['Date'] = f['Date'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_pr['C*V'] = train_pr['Volume']*train_pr['Close']
    #先粘上股票信息，带入了市场部门名称，但需要注意stock表里有生效日期，如何处理？
    train_pr = train_pr.merge(stock_list[['SecuritiesCode','IssuedShares','Section','33SectorCode','17SectorCode']],how='left',on='SecuritiesCode')
    #换手率直接加，成交量/发行股份
    train_pr['Turnover'] = train_pr['Volume']/train_pr['IssuedShares']
    
    train_trd = train_trd.rename(columns={'Date':'trd_Date'})
    train_trd['StartDate'] = train_trd['StartDate'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_trd['EndDate'] = train_trd['EndDate'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_fin['DisclosedDate'] = train_fin['DisclosedDate'].apply(lambda x :np.nan if pd.isna(x) else int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_fin['CurrentPeriodEndDate'] = train_fin['CurrentPeriodEndDate'].apply(lambda x :np.nan if pd.isna(x) else int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_fin['CurrentFiscalYearStartDate'] = train_fin['CurrentFiscalYearStartDate'].apply(lambda x :np.nan if pd.isna(x) else int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    for col in train_fin.columns:
        try:train_fin[col] = train_fin[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)    #将销售额等转换为浮点数
        except:pass
    
    #根据trade制作一个df，把【startDate，endDate】展开作index，然后值就是trd_Date
    trd_Date = {}
    for i in range(train_trd.shape[0]):
        index0 = list(range(train_trd.loc[i,'StartDate'],train_trd.loc[i,'EndDate']+1))
        value0 = train_trd.loc[i,'trd_Date']
        for j in index0:trd_Date[j]=value0
    def trds_Date(x):
        try:return trd_Date[x]
        except:return np.nan
    train_pr['trd_Date'] = train_pr['Date'].apply(lambda x:trds_Date(x))
    #price到trade表的市场映射dict，只包含2000支股票对应三大市场
    section_dict = {'First Section (Domestic)':'Prime Market (First Section)', 'Second Section(Domestic)':'Standard Market (Second Section)', 
        'Mothers (Domestic)':'Growth Market (Mothers/JASDAQ)','JASDAQ(Standard / Domestic)':'Growth Market (Mothers/JASDAQ)',
        'JASDAQ(Growth/Domestic)':'Growth Market (Mothers/JASDAQ)'}
    #再对市场部门名称 进行映射后与trade来merge
    train_pr['Section'] = train_pr['Section'].apply(lambda x:section_dict[x])
    train_trd = train_trd.merge(train_pr.groupby(['Section','trd_Date'])["C*V","Target"].agg("mean").reset_index(), how='left',on=['Section','trd_Date']).drop(columns=['StartDate','EndDate'])  #只留下市场各机构销售额之类的

    #接下来按月份粘上fin表, 先预处理
    fea_fin = ['NetSales','OperatingProfit','OrdinaryProfit','Profit','EarningsPerShare','TotalAssets','Equity','EquityToAssetRatio','BookValuePerShare','AverageNumberOfShares']   #若这些重要信息全为nan，则丢掉
    train_fin = train_fin.loc[train_fin[fea_fin].dropna(how='all').index].drop(columns=['DisclosureNumber','Date','DateCode','DisclosedTime','DisclosedUnixTime','TypeOfDocument','CurrentFiscalYearEndDate'])
    train_fin = train_fin.groupby(['SecuritiesCode','CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).agg("max").reset_index()   #去重，mean可能使得文件公布日期变化
    def month_add(ym0,nm):
        assert len(str(ym0))==6
        ym0 = ym0 + 100*(nm//12) + nm%12  #年月+月份数 的加减法，//为向下取整
        g = int(str(ym0)[-2:])
        if g//12==1:ym0 = ym0-12+100
        return ym0
    #1Q就是0-2，2Q是3-5，3Q是6-8，4Q是9-11，5Q是12-14，FY则需要根据前一个判定+多少
    #将fin表里'CurrentPeriodEndDate'列只取年月，然后扩展为 3 列【0,-1,-2】，接着在data里新建一列只取年月
    train_fin['CurrentPeriodEndDate'] = train_fin['CurrentPeriodEndDate'].apply(lambda x:int(str(x)[:6]) )
    train_pr['CurrentPeriodEndDate'] = train_pr['Date'].apply(lambda x:int(str(x)[:6]) )
    train_fin1 = copy.deepcopy(train_fin)
    train_fin1['CurrentPeriodEndDate'] = train_fin1['CurrentPeriodEndDate'].apply(lambda x:month_add(x,-1))
    train_fin = pd.concat((train_fin,train_fin1))
    train_fin1['CurrentPeriodEndDate'] = train_fin1['CurrentPeriodEndDate'].apply(lambda x:month_add(x,-1))
    train_fin = train_fin.merge(train_pr.groupby(['SecuritiesCode','CurrentPeriodEndDate'])["Close","Target"].agg("mean").reset_index(), how='left',on=['SecuritiesCode','CurrentPeriodEndDate'])
    df = train_fin.groupby(['SecuritiesCode','CurrentFiscalYearStartDate','TypeOfCurrentPeriod'])["Close","Target"].agg("mean").reset_index()
    train_fin = train_fin1.merge(df, how='left',on=['SecuritiesCode','CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).drop(columns=['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'])
    train_fin = train_fin.merge(train_pr[['SecuritiesCode','Section']].drop_duplicates(), how='left',on=['SecuritiesCode'])
    def process(a):
        for col in ['NetSales','OperatingProfit','OrdinaryProfit','Profit']:
            new = a[col].copy()
            a[col] = a[col]-a[col].shift(1)
            ind = a[a['TypeOfCurrentPeriod']=="1Q"].index
            a.loc[ind,col] = new[ind]
        a["CMA"] = (a["TotalAssets"]-a["TotalAssets"].shift(1))/a["TotalAssets"].shift(1)
        a["SGR"] = (a["NetSales"]-a["NetSales"].shift(1))/a["NetSales"].shift(1)   #shift(1)是当季度-上季度/上季度
        a["PGR"] = (a["Profit"]-a["Profit"].shift(1))/a["Profit"].shift(1)
        a["C_target"] = (a["CMV"].shift(-1)-a["CMV"])/a["CMV"]  #未来一个季度
        return a
    train_fin['CMV'] = train_fin["Close"]*train_fin["AverageNumberOfShares"]
    train_fin = train_fin.groupby('SecuritiesCode').apply(lambda x:process(x) ).reset_index(drop=True)
    #只留下了销售额等重要信息
    
    #相关因子衍生
    train_fin['ROA'] = train_fin["Profit"]/train_fin["TotalAssets"]  #资产回报率
    train_fin['RMW'] = train_fin["Profit"]/train_fin["Equity"]
    train_fin['GPM'] = train_fin["Profit"]/train_fin["NetSales"]   #销售毛利率
    train_fin['BookValuePerShare'] = train_fin['Equity']/train_fin['AverageNumberOfShares']
    series = train_fin['BookValuePerShare']/(train_fin['Close']+1e-9 )
    series.rename("BM",inplace=True)
    train_fin = train_fin.merge(series,left_index=True,right_index=True)  #账面市值比
    train_fin = train_fin.merge(stock_list[['SecuritiesCode','33SectorCode','33SectorName','17SectorCode','17SectorName']],how='left',on='SecuritiesCode')
    #按Date来groupby后用17SectorName来计算市场份额="NetSales"/"NetSales"sum
    def add_sector(x):
        df = x.groupby("17SectorName")["NetSales"].agg("sum")
        x["SOM"] = x.apply(lambda x:x["NetSales"]/df[x["17SectorName"]], axis = 1) #市场份额
        return x
    train_fin = train_fin.groupby('CurrentPeriodEndDate').apply(lambda x:add_sector(x)).reset_index(drop=True)
    
    '''
    series = train_fin['Close']/(train_fin['EarningsPerShare']+1e-9 )
    series.rename("PE",inplace=True)
    train_fin = train_fin.merge(series,left_index=True,right_index=True)  #PE
    series = train_fin['Close']/(train_fin['BookValuePerShare']+1e-9 )
    series.rename("PB",inplace=True)
    train_fin = train_fin.merge(series,left_index=True,right_index=True)  #PB
    series = train_fin['Close']*train_fin['AverageNumberOfShares']/(train_fin['NetSales']+1e-9 )
    series.rename("PS",inplace=True)
    train_fin = train_fin.merge(series,left_index=True,right_index=True)  #PS'''
    
    return train_pr,train_fin,train_trd

#粘上新增的财报、市场信息
valid_pr,valid_fin,valid_trd = pd.concat((train_pr,valid_pr)).reset_index(drop=True),pd.concat((train_fin,valid_fin)).reset_index(drop=True),pd.concat((train_trd,valid_trd)).reset_index(drop=True)
train_xy,_,_ = process(train_fin,train_op,train_pr,train_sec_pr,train_trd )
valid_xy,df_fin,df_trd = process(valid_fin,valid_op,valid_pr,valid_sec_pr,valid_trd )
valid_xy = valid_xy[valid_xy["Date"]>train_xy["Date"].max()].reset_index(drop=True)
#valid_xy = valid_xy[valid_xy["Date"]<20220300].reset_index(drop=True)
df_fin = df_fin.sort_values('CurrentPeriodEndDate').drop_duplicates(['SecuritiesCode','CurrentPeriodEndDate'],keep="last").reset_index(drop=True)  #排序、去重
def recall(df,sect=True):   #【总资产增长率CMA、销售增长率SGR、利润增长率PGR、资产回报率ROA、销售毛利率GPM、市场份额SOM、】
    # print(len(set(df['SecuritiesCode'].unique().tolist()) & Securities_Code_L))
    if sect==False:  #没有交错
        #通过总资产选取
        Equity_P = df[df["Equity"]>df["Equity"].quantile([0.97])[0.97]]["SecuritiesCode"].unique().tolist()
        Equity_P = set(Equity_P) & Securities_Code_L
        Equity_N = df[df["Equity"]<df["Equity"].quantile([0.1])[0.1]]["SecuritiesCode"].unique().tolist()
        Equity_N = set(Equity_N) & Securities_Code_L
        print("通过股东权益Equity选出 正负:",len(Equity_P),len(Equity_N))

        Assets_P = df[df["TotalAssets"]>df["TotalAssets"].quantile([0.95])[0.95]]["SecuritiesCode"].unique().tolist()
        Assets_P = set(Assets_P) & Securities_Code_L
        Assets_N = df[df["TotalAssets"]<df["TotalAssets"].quantile([0.2])[0.2]]["SecuritiesCode"].unique().tolist()
        Assets_N = set(Assets_N) & Securities_Code_L
        print("通过总资产TotalAssets选出 正负:",len(Assets_P),len(Assets_N))
        res_P = Equity_P | Assets_P
        res_N = Equity_N | Assets_N
    else:  #交错很多，可能要增添交互条件
        # CMA_P = df[df["CMA"]>df["CMA"].quantile([0.95])[0.95]]["SecuritiesCode"].unique().tolist()
        # CMA_P = set(CMA_P) & Securities_Code_L
        # CMA_N = df[df["CMA"]<df["CMA"].quantile([0.1])[0.1]]["SecuritiesCode"].unique().tolist()
        # CMA_N = set(CMA_N) & Securities_Code_L
        # print("通过总资产增长率CMA选出 正负:",len(CMA_P),len(CMA_N))

        # SGR_P = df[df["SGR"]>df["SGR"].quantile([0.85])[0.85]]["SecuritiesCode"].unique().tolist()
        # SGR_P = set(SGR_P) & Securities_Code_L
        # SGR_N = df[df["SGR"]<df["SGR"].quantile([0.1])[0.1]]["SecuritiesCode"].unique().tolist()
        # SGR_N = set(SGR_N) & Securities_Code_L
        # print("通过销售增长率SGR选出 正负:",len(SGR_P),len(SGR_N))

        # PGR_P = df[df["PGR"]>df["PGR"].quantile([0.85])[0.85]]["SecuritiesCode"].unique().tolist()
        # PGR_P = set(PGR_P) & Securities_Code_L
        # PGR_N = df[df["PGR"]<df["PGR"].quantile([0.2])[0.2]]["SecuritiesCode"].unique().tolist()
        # PGR_N = set(PGR_N) & Securities_Code_L
        # print("通过利润增长率PGR选出 正负:",len(PGR_P),len(PGR_N))

        ROA_P = df[df["ROA"]>df["ROA"].quantile([0.9])[0.9]]["SecuritiesCode"].unique().tolist()
        ROA_P = set(ROA_P) & Securities_Code_L
        ROA_N = df[df["ROA"]<df["ROA"].quantile([0.2])[0.2]]["SecuritiesCode"].unique().tolist()
        ROA_N = set(ROA_N) & Securities_Code_L
        print("通过资产回报率ROA选出 正负:",len(ROA_P),len(ROA_N))

        GPM_P = df[df["GPM"]>df["GPM"].quantile([0.8])[0.8]]["SecuritiesCode"].unique().tolist()
        GPM_P = set(GPM_P) & Securities_Code_L
        GPM_N = df[df["GPM"]<df["GPM"].quantile([0.25])[0.25]]["SecuritiesCode"].unique().tolist()
        GPM_N = set(GPM_N) & Securities_Code_L
        print("通过销售毛利率GPM选出 正负:",len(GPM_P),len(GPM_N))

        SOM_P = df[df["SOM"]>df["SOM"].quantile([0.85])[0.85]]["SecuritiesCode"].unique().tolist()
        SOM_P = set(SOM_P) & Securities_Code_L
        SOM_N = df[df["SOM"]<df["SOM"].quantile([0.3])[0.3]]["SecuritiesCode"].unique().tolist()
        SOM_N = set(SOM_N) & Securities_Code_L
        print("通过市场份额SOM选出 正负:",len(SOM_P),len(SOM_N))

        res_P = ROA_P | GPM_P | SOM_P   #CMA_P | SGR_P | PGR_P | 
        res_N = ROA_N | GPM_N | SOM_N
    print("选出 正负:",len(res_P),len(res_N))
    res_PN = res_P&res_N
    print("正负里都有的交错项:",len(res_PN))   #可能存在既要买入又要做空的情况
    # res_P = res_P-res_PN

    return res_P,res_N
# 直接先对valid初筛
#只留下最后一个季度，注意此处月份需+2为原月份
df_fin = df_fin[df_fin['CurrentPeriodEndDate']>=202107].sort_values('CurrentPeriodEndDate').drop_duplicates(['SecuritiesCode'],keep="last").reset_index(drop=True)
P_L,N_L = recall(df_fin,False)
for _,df0 in df_fin.groupby("17SectorName"):
    #(df0[0],df0[1])元组形式
    print("17SectorName:",_)
    P_L0,N_L0 = recall(df0)
    P_L,N_L = P_L|P_L0, N_L|N_L0
print("选出 正负:",len(P_L),len(N_L))
PN_L = P_L&N_L
print("正负里都有的交错项:",len(PN_L))
# P_L,N_L = Securities_Code_L,Securities_Code_L

for col in ['AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag']:
    train_xy[col] = train_xy[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)
    valid_xy[col] = valid_xy[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)
print("train维度:",train_xy.shape)
time_now = time.time()

#返回按['SecuritiesCode','Date']降序排列的df
print("计算MACD")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:MACD_add(x)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:MACD_add(x)).reset_index(drop=True)
print("train维度:",train_xy.shape)
print("引入K线")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,1)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,1)).reset_index(drop=True)
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,3)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,3)).reset_index(drop=True)
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,5)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,5)).reset_index(drop=True)
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,9)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,9)).reset_index(drop=True)
print("train维度:",train_xy.shape)
print("引入乖离率、量比、RSI")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:other_add(x)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:other_add(x)).reset_index(drop=True)
print("train维度:",train_xy.shape)
print("计算KDJ")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:KDJ_add(x)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:KDJ_add(x)).reset_index(drop=True)
print("train维度:",train_xy.shape)
print("添加因子耗时min:",(time.time()-time_now)/60)

def custprod_1group(kstr,dstr,listgroup,keystr,actlist):   #一次只能有一个keystr，做之前一个月时间的统计特征
    df = data_all[kstr]
    _df = pd.DataFrame()
    for now in tqdm.tqdm(df['Date'].unique()):
        _df0 = df[df[dstr].apply(lambda x:now-301<x<=now)].groupby(listgroup)[keystr].agg(actlist).reset_index()
        _df0['Date'] = now
        _df = pd.concat((_df,_df0)).reset_index(drop=True)
    _df.columns = listgroup+[kstr+'_' + keystr+'_1m_'+i for i in actlist]+['Date']
    return _df,_df.columns[1]

test_now = 20211207
#valid_xy = valid_xy[valid_xy['Date']<=test_now]   #按日期取出valid范围

print("train里最大日期:",train_xy['Date'].max(),train_xy.shape)
'''
data_all = {}
data_all['price'] = train_xy
TS_df,col = custprod_1group('price','Date',["SecuritiesCode"],'Target',['mean'])
train_xy = train_xy.merge(TS_df,how='left',on=["SecuritiesCode","Date"])
#train_xy = train_xy[train_xy['Date']>20180104 ]
print(train_xy.shape)
#对test数据不用按Date来粘
TS_df = train_xy[train_xy["Date"].apply(lambda x:test_now-301<x<test_now-1)].groupby(["SecuritiesCode"])["Target"].agg("mean").reset_index()
TS_df = pd.DataFrame(data=TS_df["Target"].values,index=TS_df["SecuritiesCode"].values,columns=[col])
def get_avg(_id_):
    return TS_df.loc[_id_]
valid_xy[col] = valid_xy["SecuritiesCode"].apply(get_avg)   #线上评分时也用此方式
'''
print("valid是否符合sharp ratio函数规格",valid_xy.shape,valid_xy["SecuritiesCode"].nunique())

seq,epochs,lr = 5+2,20,3e-4
train_xy = train_xy[train_xy["Target"].notna()].reset_index(drop=True)
Drop_fea = [col for col in train_xy.columns if train_xy[col].dtype=='object'] + ["Open","Low","High",'Volume','AdjustmentFactor','SupervisionFlag','ExpectedDividend']
print("Drop_fea:",Drop_fea)
train_xy,valid_xy = train_xy.drop(columns=Drop_fea),valid_xy.drop(columns=Drop_fea)
fea_L = train_xy.shape[1]-3
print("fea_L:",fea_L)
fea = ["Close"]+train_xy.drop(columns=["Close"]).columns.tolist()  #Close放第一个
train_xy,valid_xy = train_xy[fea],valid_xy[fea]
print(train_xy.columns)

for col in train_xy.columns:  #归一化方法也有待斟酌
    maxmin_Close = train_xy[col].quantile([0.01,0.99])
    if col in {'Date','SecuritiesCode',"Target"} or train_xy[col].nunique()<10 or maxmin_Close[0.99]==maxmin_Close[0.01]:continue
    train_xy[col],valid_xy[col] = (train_xy[col]-maxmin_Close[0.01])/(maxmin_Close[0.99]-maxmin_Close[0.01]), (valid_xy[col]-maxmin_Close[0.01])/(maxmin_Close[0.99]-maxmin_Close[0.01])
train_xy,valid_xy = train_xy.fillna(0),valid_xy.fillna(0)
#from pandarallel import pandarallel   #只有在Linux下运行
#pandarallel.initialize()
def get_window(df,s):  #输入一个2000的df
    #padding前面seq-1个0
    df0 = copy.deepcopy(df)
    df = df[["Date",'SecuritiesCode']]  #实现把这两个放第一个，target放最后
    df0 = pd.concat((df0, pd.DataFrame(data = np.zeros([s-1,df0.shape[1]]),columns = df0.columns) ))
    df0 = df0.sort_values("Date").reset_index(drop=True)
    dfx,dft = df0.drop(columns=["Target",'SecuritiesCode',"Date"]),df0["Target"]
    df_fea,df_Target = [x.values.reshape(-1).tolist() for x in dfx.rolling(s)][s-1:], [x.values.reshape(-1).tolist() for x in dft.rolling(s)][s-1:]   #也可以摊平，然后转tensor的时候再reshape
    df_fea,df_Target = pd.DataFrame(data=df_fea,columns=[f"fea{i}" for i in range(s*fea_L )],index=df.index), pd.DataFrame(data=df_Target,columns=[f"Target{i}" for i in range(s-1)]+["Target"],index=df.index)
    # df_fea = reduce_mem(df_fea)
    df = pd.concat((df,df_fea,df_Target),axis=1)  #注意index的对应
    #df["fea"],df["Target"] = [str(x.values.tolist()) for x in dfx.rolling(s)][s-1:], [str(x.values.tolist()) for x in dft.rolling(s)][s-1:]  #要调用需要json.loads(x)
    return df
#滑窗成时序特征形式的df，parallel_
train_xy,valid_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:get_window(x.sort_values('Date'),seq)).reset_index(drop=True), valid_xy.groupby('SecuritiesCode').apply(lambda x:get_window(x.sort_values('Date'),seq-2)).reset_index(drop=True)
print("缺失值检查",train_xy.isnull().sum().sum(),valid_xy.isnull().sum().sum())
#然后按date、target排序后分成1000+组数据丢进去用listmle训练
train_xy,valid_xy = train_xy.groupby('Date').apply(lambda x:x.sort_values("Target",ascending=False)).reset_index(drop=True),valid_xy.groupby('Date').apply(lambda x:x.sort_values("Target",ascending=False)).reset_index(drop=True)

#后面按g_train喂进模型即可
g_train = train_xy.groupby(['Date'], as_index=False).count()['SecuritiesCode'].values
print("时序滑窗耗时min:",(time.time()-time_now)/60)
lastday = valid_xy['Date'].max()
print("最后一天的日期",lastday,train_xy.shape)
print("train里特征:",train_xy.columns)
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    输入是一个每天都包含2000支股票的rank+target的df，且如果仅有一天会std=0导致nan
    calc_spread_return_sharpe(df[['Date','SecuritiesCode','Rank','Target']], 200, 2)
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        #weights=[2,1]之间的200个点，计算前200个的purchase，后200个的short
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by="Rank")['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by="Rank", ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    if df['Date'].nunique()==1:sharpe_ratio = buf.values[0]
    else:sharpe_ratio = buf.mean() / (1e-9+buf.std())
    return sharpe_ratio
class model(nn.Module):
    def __init__(self):
        super().__init__()
        #train_xy.shape[1]-3
        self.Dense0 = nn.Linear(fea_L-1,1)  #(seq-2)
        self.LSTM = nn.LSTM(fea_L-1,1)
    def forward(self,X ):   #输入一个【bz，seq，dim】,如果要序列，注意应该从下标seq-1处开始往前走n个序列【bz，-n:，dim】
        #X = X.permute(1,0,2)
        #out = self.LSTM(X)[0].permute(1,0,2).flatten(1)[:,-1]
        out = nn.Sigmoid()(self.Dense0(X[:,-1,:]).flatten(0))
        return out
def add_rank(x,ranks):
    x = x.sort_values(by="Prediction", ascending=False)
    x['Rank'] = range(ranks,ranks+x.shape[0])
    return x
def get_res(model1,valid_xy,L,ranks):  #L为P_L或N_L，P_L的ranks=0，N_L的ranks=len(P_L)+1
    valid_xy = valid_xy[valid_xy["SecuritiesCode"].apply(lambda x:x in L)].reset_index(drop=True)
    g_eval = valid_xy.groupby(['Date'], as_index=False).count()['SecuritiesCode'].values
    y_pred1,Target = [],[]
    loss0,seq0 = 0,seq-2
    with torch.no_grad():
        for i in range(len(g_eval)):
            g_ = g_eval[:i].sum()
            batch = valid_xy.iloc[g_:g_+g_eval[i]].copy()
            y_d = [i for i in range(1,seq0)]+[seq0]*(g_eval[i]-2*seq0+2)+[seq0-i for i in range(1,seq0)]
            #batch['fea'],batch['Target'] = batch['fea'].apply(json.loads), batch['Target'].apply(json.loads)
            X,T = batch[[f"fea{i}" for i in range(seq0*fea_L )]].values.reshape(-1,seq0,fea_L ).tolist(), batch[[f"Target{i}" for i in range(seq0-1)]+["Target"]].values.reshape(-1,seq0).tolist()
            X,T = torch.tensor(X,dtype=torch.float32).to(device), torch.tensor(T,dtype=torch.float32).to(device)
            X1 = X[:,:,1:]
            y_pred11 = np.zeros([g_eval[i],])
            out1 = model1(X1)
            loss = ListMLE_loss(out1, T[:,-1] )
            loss0 += loss.mean().item()
            out1 = out1.detach().cpu().numpy()
            #for j in range(g_eval[i]-seq0):y_pred11[j: seq0+j] += out1[j,:]
            #y_pred11 = y_pred11/y_d
            y_pred1 += out1.tolist()
            torch.cuda.empty_cache()
    print("valid_loss:", loss0/(i+1) )
    
    valid_1 = valid_xy["Date"].copy().to_frame()
    valid_1["Prediction"] = y_pred1
    valid_1["Target"] = valid_xy["Target"].copy()
    valid_1 = valid_1[["Date","Prediction","Target"]].groupby("Date").apply(lambda x:add_rank(x,ranks) ).reset_index(drop=True)
    return valid_1

def evaluation_valid(model1,valid_xy):
    df_P_L1 = get_res(model1,valid_xy,P_L,0)
    df_N_L1 = get_res(model1,valid_xy,N_L,len(P_L)+1)
    valid_ = pd.concat((df_P_L1,df_N_L1)).reset_index(drop=True)
    score = calc_spread_return_sharpe(valid_, 200, 2)
    return score

def ListMLE_loss(Y,T ):
    #_,indices = Y.sort(descending=True,dim=1)  #未提前排序时，需要实时排序一下
    #Y = Y.gather(dim=1,index=indices)
    Y_e = Y.exp()
    Y_e = Y_e.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])  #实现从尾递加到头部
    res = torch.log(Y_e)-Y
    return res.mean(dim=-1)

b_score = 0
for random_seed in [42]:
    seed_torch(random_seed)
    #.sample(frac=1,random_state=random_seed)
    model = model().to(device)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-30)
    #model.train()
    print('================     random seed {}        ==============='.format(random_seed))
    for epoch in range(epochs):
        for i in tqdm.tqdm(range(len(g_train)),desc=f'epoch{epoch+1}' ):
            g_ = g_train[:i].sum()
            batch = train_xy.iloc[g_:g_+g_train[i]].copy()   #不要前面带0的参与训练
            #batch['fea'],batch['Target'] = batch['fea'].apply(json.loads), batch['Target'].apply(json.loads)
            X,T = batch[[f"fea{i}" for i in range(seq*fea_L )]].values.reshape(-1,seq,fea_L ).tolist(), batch[[f"Target{i}" for i in range(seq-1)]+["Target"]].values.reshape(-1,seq).tolist()
            X,T = torch.tensor(X,dtype=torch.float32).to(device), torch.tensor(T,dtype=torch.float32).to(device)
            X1,T1 = X[:,:,1:],T[:,-1]
            out = model(X1)
            loss = ListMLE_loss(out,T1 )#nn.MSELoss()
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            torch.cuda.empty_cache()
        print("train_loss:", loss.item() )
        score = evaluation_valid(model,valid_xy)
        print("valid_score:",score)
        if score>b_score:
            b_score = score
            torch.save(model.state_dict() , f"model{random_seed}.pth")  #model.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))

#先对userid排序后，即可传入从开头到后面每隔多少个数据进行rank的损失限定
#此处则按Date进行排序，然后每个传入2000支股票即可

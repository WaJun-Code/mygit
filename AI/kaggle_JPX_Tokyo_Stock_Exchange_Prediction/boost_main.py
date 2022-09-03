from utils import *
import numpy as np
import pandas as pd
import joblib
'''
train_xy.Date = pd.to_datetime(train_xy.Date)
train_xy['Date'] = train_xy['Date'].dt.strftime("%Y%m%d").astype(int)
'''
from lightgbm import LGBMRegressor,LGBMRanker
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import warnings,tqdm,re,copy
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
def month_add(ym0,nm):
    assert len(str(ym0))==6
    ym0 = ym0 + 100*(nm//12) + nm%12  #年月+月份数 的加减法，//为向下取整
    g = int(str(ym0)[-2:])
    if g//12==1:ym0 = ym0-12+100
    return ym0
def process(train_fin,train_op,train_pr,train_sec_pr,train_trd ):
    for f in [train_fin,train_op,train_pr,train_sec_pr,train_trd ]:
        f['Date'] = f['Date'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_pr['C*V'] = train_pr['Volume']*train_pr['Close']
    #先粘上股票信息，带入了市场部门名称，但需要注意stock表里有生效日期，如何处理？
    train_pr = train_pr.merge(stock_list[['SecuritiesCode','IssuedShares','Section','33SectorCode','17SectorCode']],how='left',on='SecuritiesCode')
    #换手率直接加，成交量/发行股份
    train_pr['Turnover'] = train_pr['Volume']/train_pr['IssuedShares']
    
    train_trd['StartDate'] = train_trd['StartDate'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_trd['EndDate'] = train_trd['EndDate'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_fin['CurrentFiscalYearStartDate'] = train_fin['CurrentFiscalYearStartDate'].apply(lambda x :np.nan if pd.isna(x) else int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    for col in train_fin.columns:
        try:train_fin[col] = train_fin[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)    #将销售额等转换为浮点数
        except:pass
    
    #price到trade表的市场映射dict，只包含2000支股票对应三大市场
    section_dict = {'First Section (Domestic)':'Prime Market (First Section)', 'Second Section(Domestic)':'Standard Market (Second Section)', 
        'Mothers (Domestic)':'Growth Market (Mothers/JASDAQ)','JASDAQ(Standard / Domestic)':'Growth Market (Mothers/JASDAQ)',
        'JASDAQ(Growth/Domestic)':'Growth Market (Mothers/JASDAQ)'}
    #再对市场部门名称 进行映射后与trade来merge
    train_pr['Section'] = train_pr['Section'].apply(lambda x:section_dict[x])

    fea_fin = ['NetSales','OperatingProfit','OrdinaryProfit','Profit','EarningsPerShare','TotalAssets','Equity','EquityToAssetRatio','BookValuePerShare','AverageNumberOfShares']   #若这些重要信息全为nan，则丢掉
    train_fin = train_fin.loc[train_fin[fea_fin].dropna(how='all').index].drop(columns=['DisclosureNumber','DisclosedDate','DateCode','DisclosedTime','DisclosedUnixTime','TypeOfDocument','CurrentPeriodEndDate','CurrentFiscalYearEndDate'])
    train_fin = train_fin.groupby(['SecuritiesCode','CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).agg("max").reset_index()   #去重，mean可能使得文件公布日期变化

    #1Q就是0-2，2Q是3-5，3Q是6-8，4Q是9-11，5Q是12-14，FY则需要根据前一个判定+多少
    def process(a):
        for col in ['NetSales','OperatingProfit','OrdinaryProfit','Profit']:
            new = a[col].copy()
            a[col] = a[col]-a[col].shift(1)
            ind = a[a['TypeOfCurrentPeriod']=="1Q"].index
            a.loc[ind,col] = new[ind]
        a["CMA"] = (a["TotalAssets"]-a["TotalAssets"].shift(1))/a["TotalAssets"].shift(1)
        a["SGR"] = (a["NetSales"]-a["NetSales"].shift(1))/a["NetSales"].shift(1)   #shift(1)是当季度-上季度/上季度
        a["PGR"] = (a["Profit"]-a["Profit"].shift(1))/a["Profit"].shift(1)
        return a
    train_fin = train_fin.groupby('SecuritiesCode').apply(lambda x:process(x) ).reset_index(drop=True)
    #只留下了销售额等重要信息
    
    #相关因子衍生
    train_fin['ROA'] = train_fin["Profit"]/train_fin["TotalAssets"]  #资产回报率
    train_fin['RMW'] = train_fin["Profit"]/train_fin["Equity"]
    train_fin['GPM'] = train_fin["Profit"]/train_fin["NetSales"]   #销售毛利率
    train_fin['BookValuePerShare'] = train_fin['Equity']/train_fin['AverageNumberOfShares']
    train_fin = train_fin.merge(stock_list[['SecuritiesCode','33SectorCode','33SectorName','17SectorCode','17SectorName']],how='left',on='SecuritiesCode')
    #按Date来groupby后用17SectorName来计算市场份额="NetSales"/"NetSales"sum
    def add_sector(x):
        df = x.groupby("17SectorName")["NetSales"].agg("sum")
        x["SOM"] = x.apply(lambda x:x["NetSales"]/df[x["17SectorName"]], axis = 1) #市场份额
        return x
    train_fin = train_fin.groupby(['CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).apply(lambda x:add_sector(x)).reset_index(drop=True)
    '''
    series = train_fin['BookValuePerShare']/(train_fin['Close']+1e-9 )
    series.rename("BM",inplace=True)
    train_fin = train_fin.merge(series,left_index=True,right_index=True)  #账面市值比
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
train_xy,train_fin,train_trd = process(train_fin,train_op,train_pr,train_sec_pr,train_trd )
valid_xy,valid_fin,valid_trd = process(valid_fin,valid_op,valid_pr,valid_sec_pr,valid_trd )
valid_xy = valid_xy[valid_xy["Date"]>train_xy["Date"].max()].reset_index(drop=True)
#valid_xy = valid_xy[valid_xy["Date"]<20220300].reset_index(drop=True)
valid_fin = valid_fin.sort_values(['CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).drop_duplicates(['SecuritiesCode','CurrentFiscalYearStartDate','TypeOfCurrentPeriod'],keep="last").reset_index(drop=True)  #排序、去重
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
valid_fin = valid_fin.sort_values(['CurrentFiscalYearStartDate','TypeOfCurrentPeriod']).drop_duplicates(['SecuritiesCode'],keep="last").reset_index(drop=True)
print(valid_fin.shape[0],valid_fin['SecuritiesCode'].nunique())
P_L,N_L = recall(valid_fin,False)
for _,df0 in valid_fin.groupby("17SectorName"):
    #(df0[0],df0[1])元组形式
    print("17SectorName:",_)
    P_L0,N_L0 = recall(df0)
    P_L,N_L = P_L|P_L0, N_L|N_L0
print("选出 正负:",len(P_L),len(N_L))
PN_L = P_L&N_L
print("正负里都有的交错项:",len(PN_L))
P_L,N_L = Securities_Code_L,Securities_Code_L

for col in ['AdjustmentFactor', 'ExpectedDividend', 'SupervisionFlag']:
    train_xy[col] = train_xy[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)
    valid_xy[col] = valid_xy[col].apply(lambda x:np.nan if x=='－' else x).astype(np.float32)
print("train维度:",train_xy.shape)
time_now = time.time()

#返回按['SecuritiesCode','Date']降序排列的df
print("引入K线")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,1)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,1)).reset_index(drop=True)
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,3)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,3)).reset_index(drop=True)
# train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,5)).reset_index(drop=True)
# valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,5)).reset_index(drop=True)
# train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,9)).reset_index(drop=True)
# valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:K_add(x,9)).reset_index(drop=True)
print("train维度:",train_xy.shape)
print("引入乖离率、量比、RSI")
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:other_add(x)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:other_add(x)).reset_index(drop=True)
print("train维度:",train_xy.shape)
# print("计算MACD")
# train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:MACD_add(x)).reset_index(drop=True)
# valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:MACD_add(x)).reset_index(drop=True)
# print("train维度:",train_xy.shape)
# print("计算KDJ")
# train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:KDJ_add(x)).reset_index(drop=True)
# valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:KDJ_add(x)).reset_index(drop=True)
# print("train维度:",train_xy.shape)
print("添加因子耗时min:",(time.time()-time_now)/60)

test_now = 20211207
#valid_xy = valid_xy[valid_xy['Date']<=test_now]   #按日期取出valid范围

print("train里最大日期:",train_xy['Date'].max(),train_xy.shape)
print("valid是否符合sharp ratio函数规格",valid_xy.shape,valid_xy["SecuritiesCode"].nunique())

seq = 7
train_xy = train_xy[train_xy["Target"].notna()].reset_index(drop=True)
print("valid里的标签nan数量:",valid_xy["Target"].isnull().sum())
valid_xy = valid_xy[valid_xy["Target"].notna()].reset_index(drop=True)

Drop_fea = [col for col in train_xy.columns if train_xy[col].dtype=='object' and col!="Section"] + ["Close","Open","Low","High",'Volume','AdjustmentFactor','SupervisionFlag','ExpectedDividend', 'C*V', 'IssuedShares']
print("Drop_fea:",Drop_fea)
train_xy,valid_xy = train_xy.drop(columns=Drop_fea),valid_xy.drop(columns=Drop_fea)
print(train_xy.columns)

def get_window(df,s):  #输入一个2000的df
    #padding前面seq-1个0
    df0 = copy.deepcopy(df)
    df = df[["Date",'SecuritiesCode',"Section"]]  #实现把这两个放第一个，target放最后
    df0 = pd.concat((df0, pd.DataFrame(data = np.zeros([s-1,df0.shape[1]]),columns = df0.columns) ))
    df0 = df0.sort_values("Date").reset_index(drop=True)
    dfx,dft = df0.drop(columns=["Target",'SecuritiesCode',"Section","Date"]),df0["Target"]
    df_fea,df_Target = [x.values.reshape(-1).tolist() for x in dfx.rolling(s)][s-1:], [x.values.reshape(-1).tolist() for x in dft.rolling(s)][s-1:]   #也可以摊平，然后转tensor的时候再reshape
    df_fea,df_Target = pd.DataFrame(data=df_fea,columns=[f"fea{i}" for i in range(s*fea_L )],index=df.index), pd.DataFrame(data=df_Target,columns=[f"Target{i}" for i in range(s-1)]+["Target"],index=df.index)
    # df_fea = reduce_mem(df_fea)
    df = pd.concat((df,df_fea,df_Target),axis=1)  #注意index的对应
    #df["fea"],df["Target"] = [str(x.values.tolist()) for x in dfx.rolling(s)][s-1:], [str(x.values.tolist()) for x in dft.rolling(s)][s-1:]  #要调用需要json.loads(x)
    return df
#from pandarallel import pandarallel   #只有在Linux下运行
#pandarallel.initialize()
#滑窗成时序特征形式的df，parallel_
fea_L = train_xy.shape[1]-4
print("fea_L:",fea_L)
train_xy = train_xy.groupby('SecuritiesCode').apply(lambda x:get_window(x.sort_values('Date'),seq)).reset_index(drop=True)
valid_xy = valid_xy.groupby('SecuritiesCode').apply(lambda x:get_window(x.sort_values('Date'),seq)).reset_index(drop=True)
def day_add(yd0,nd):
    m_dict = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    assert len(str(int(yd0)))==8
    month = (yd0//100)%100
    m_ = nd//m_dict[month]
    y_ = (month + m_ -1)//12
    yd0 = yd0 + 10000*y_ + 100*(m_-12*y_) + nd%m_dict[month]  #年月日+天数 不能超过两个月60天
    return yd0
def merge_fintrd(train_xy,train_fin,train_trd):
    train_fin0,train_trd0 = copy.deepcopy(train_fin),copy.deepcopy(train_trd)
    for i in range(1,7):
        df = copy.deepcopy(train_fin0)
        df["Date"] = df["Date"].apply(lambda x:day_add(x,i))
        train_fin = pd.concat((train_fin,df)).reset_index(drop=True)
        df = copy.deepcopy(train_trd0)
        df["Date"] = df["Date"].apply(lambda x:day_add(x,i))
        train_trd = pd.concat((train_trd,df)).reset_index(drop=True)
    train_xy = train_xy.merge(train_trd.drop_duplicates(['Section','Date'],keep="last"), how='left',on=['Section','Date']).drop(columns=['StartDate','EndDate'])  #只留下市场各机构销售额之类的
    train_xy = train_xy.merge(train_fin.drop_duplicates(['SecuritiesCode','Date'],keep="last"),how='left',on=['SecuritiesCode','Date']).drop(columns=['CurrentFiscalYearStartDate','TypeOfCurrentPeriod'])
    return train_xy
train_xy,valid_xy = merge_fintrd(train_xy,train_fin,train_trd),merge_fintrd(valid_xy,valid_fin,valid_trd)
Drop_fea = ["33SectorCode", "33SectorName", "17SectorCode", "17SectorName"]
train_xy,valid_xy = train_xy.drop(columns=Drop_fea),valid_xy.drop(columns=Drop_fea)

print("缺失值检查",train_xy.isnull().sum().sum(),valid_xy.isnull().sum().sum())
#然后按date、target排序后分成1000+组数据丢进去用listmle训练
train_xy,valid_xy = train_xy.groupby('Date').apply(lambda x:x.sort_values("Target",ascending=False)).reset_index(drop=True),valid_xy.groupby('Date').apply(lambda x:x.sort_values("Target",ascending=False)).reset_index(drop=True)

#后面按g_train喂进模型即可
g_train = train_xy.groupby(['Date'], as_index=False).count()['SecuritiesCode'].values
g_eval = valid_xy.groupby(['Date'], as_index=False).count()['SecuritiesCode'].values

print("时序滑窗耗时min:",(time.time()-time_now)/60)
lastday = valid_xy['Date'].max()
print("最后一天的日期",lastday,train_xy.shape)
print("train里特征:",train_xy.columns)
'''
import optuna
def objectives(trial):
    params = {
            'num_leaves': trial.suggest_int('num_leaves', 300, 4000),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_bin': trial.suggest_int('max_bin', 2, 100),
            'learning_rate': trial.suggest_uniform('learning_rate',0, 1),
    }

    model = LGBMRegressor(**params)
    model.fit(train_x,train_y)
    #model.score(X,y)计算的R2相关系数
    valid_ = valid_xy.copy()
    valid_["Prediction"] = model.predict(valid_.drop(columns=["Target"]))
    valid_ = valid_[["Date","SecuritiesCode","Prediction","Target"]]
    score = calc_spread_return_sharpe(valid_, 200, 2)
    return score
opt = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=0))
opt.optimize(objectives, n_trials=20)
print(opt.best_params)
print(opt.best_value)
trial = opt.best_trial
params_best = dict(trial.params.items())
params_best['random_seed'] = 0
'''
random_seed = 42
lastday = valid_xy['Date'].max()
print("最后一天的日期",lastday)
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
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
def custom_loss(y_true,y_pred):  #提前按Date来groupby并排序好，存下类似g_train之类的数据，然后按g_train数目进行loss计算
    #把pred按g_train分好组
    mmax = g_train.max()
    exp_y = np.exp(y_pred)
    #先做 exp_y 对应的mask表求分母下的sum，上三角
    mask0 = []
    for i in range(len(g_train)):
        g_ = g_train[:i].sum()
        tmp = exp_y[g_:g_+g_train[i]]
        mask0 += np.triu(np.meshgrid(tmp,tmp)[0] ).sum(axis=1).tolist()  #保留上三角
    mask0 = np.array(mask0)
    grad,hess = [],[]
    for i in range(len(g_train)):
        g_ = g_train[:i].sum()
        tmp = mask0[g_:g_+g_train[i]]
        tmp = np.meshgrid(tmp,tmp)[0]
        tmp = np.divide(exp_y[g_:g_+g_train[i]].reshape(-1,1),tmp )
        tmp = np.tril(tmp)  #保留下三角
        grad += tmp.sum(axis=1).tolist()
        tmp = tmp-tmp**2
        hess += tmp.sum(axis=1).tolist()
    grad = np.array(grad)-1
    hess = np.array(hess)
    return grad,hess
def add_rank(x,ranks):
    x = x.sort_values(by="Prediction", ascending=False)
    x['Rank'] = range(ranks,ranks+x.shape[0])
    return x
def custom_lgb_eval(y_true, y_pred):  #仅用一天的来早停
    T,Y = pd.DataFrame(data=y_true,columns=["Target"]),pd.DataFrame(data=y_pred,columns=["Prediction"])
    TY = pd.concat((T,Y),axis=1)
    TY["Date"] = 0
    for i in range(len(g_eval)-1):TY.loc[ g_eval[:i+1].sum():g_eval[:i+2].sum() ,"Date"]=i+1
    TY = TY[["Date","Prediction","Target"]].groupby("Date").apply(lambda x:add_rank(x,0) ).reset_index(drop=True)
    score = calc_spread_return_sharpe(TY, 200, 2)
    return 'sharp_ratio',score, True
def evaluation_valid(valid_xy,model):
    df_P_L1 = valid_xy[valid_xy["SecuritiesCode"].apply(lambda x:x in P_L)].reset_index(drop=True)
    df_P_L1["Prediction"] = model.predict( df_P_L1.drop(columns=["Date","SecuritiesCode","Section","Target"]+[f"Target{i}" for i in range(seq-1)]).values,num_iteration=model.best_iteration_ )  #lgb换xgb需要改此处以及,ntree_limit = model.best_ntree_limit
    df_P_L1 = df_P_L1[["Date","Prediction","Target"]].groupby("Date").apply(lambda x:add_rank(x,0) ).reset_index(drop=True)
    df_N_L1 = valid_xy[valid_xy["SecuritiesCode"].apply(lambda x:x in N_L)].reset_index(drop=True)
    df_N_L1["Prediction"] = model.predict( df_N_L1.drop(columns=["Date","SecuritiesCode","Section","Target"]+[f"Target{i}" for i in range(seq-1)]).values,num_iteration=model.best_iteration_ )  #,num_iteration=model.best_iteration_
    df_N_L1 = df_N_L1[["Date","Prediction","Target"]].groupby("Date").apply(lambda x:add_rank(x,len(P_L)+1) ).reset_index(drop=True)
    valid_ = pd.concat((df_P_L1,df_N_L1)).reset_index(drop=True)
    score = calc_spread_return_sharpe(valid_, 200, 2)
    return score
def custom_xgb_eval(pred,dtrain):
    T,Y = dtrain.get_label(),pred
    T,Y = pd.DataFrame(data=T,columns=["Target"]),pd.DataFrame(data=Y,columns=["Prediction"])
    TY = pd.concat((T,Y),axis=1)
    TY["Date"] = 0
    for i in range(len(g_eval)-1):TY.loc[ g_eval[:i+1].sum():g_eval[:i+2].sum() ,"Date"]=i+1
    TY = TY[["Date","Prediction","Target"]].groupby("Date").apply(lambda x:add_rank(x,0) ).reset_index(drop=True)
    score = calc_spread_return_sharpe(TY, 200, 2)
    return 'sharp_ratio',score

train_x,train_y = train_xy.drop(columns=["Date","SecuritiesCode","Section","Target"]+[f"Target{i}" for i in range(seq-1)]),train_xy["Target"]
'''
params = {
            'booster': 'gbtree', 'objective': 'reg:squarederror',  #reg:squarederror  rank:ndcg
            #'eval_metric': ['logloss','auc'],
            'gamma': 1, 'min_child_weight': 3, 'max_depth': 8, 'learning_rate': 0.01, 'lambda': 1, 'alpha': 1,
            'subsample': 0.7, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7, 'eta': 0.05,     #0.05-0.3
            'tree_method': 'exact', 'seed': random_seed, 'nthread': 48, "silent": True
            }
watchlist = [( xgb.DMatrix(valid_xy.drop(columns=["Date","SecuritiesCode","Target"]).values, label=valid_xy["Target"].values), 'eval')]
model = xgb.train( params, xgb.DMatrix(train_x.values, label=train_y.values), evals=watchlist,feval=custom_xgb_eval,maximize=True , num_boost_round=1000, early_stopping_rounds=500, verbose_eval=20)
print("\n".join(("%s: %.2f" % x) for x in list(sorted( model.get_fscore().items(),key=lambda x:x[1],reverse=False)) ))
print("最优迭代步:",model.best_iteration,model.best_ntree_limit, model.best_score)
model.save_model(f"xgb_model{random_seed}.model")   #model = xgb.Booster(model_file="xgb_model{random_seed}.model")但此时得xgb.DMatrix()
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(model, height=0.5, ax=ax, max_num_features=32)
plt.show()
score = evaluation_valid(valid_xy,model)
print("valid_score:",score)
'''


#先对userid排序后，即可传入从开头到后面每隔多少个数据进行rank的损失限定
#此处则按Date进行排序，然后每个传入2000支股票即可
#model = LGBMRanker(boosting_type='gbdt',objective='lambdarank', num_leaves=512, n_estimators=5000, reg_alpha=0.1, reg_lambda=0.1, max_depth=8, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
#                                learning_rate=0.01, min_child_weight=30, random_state=random_seed, n_jobs=-1)
#model.fit(train_x,train_y, group=g_train, eval_set=[watchlist], eval_group=[g_eval], eval_metric = custom_lgb_eval, early_stopping_rounds=2000, verbose=20)

watchlist = (valid_xy.drop(columns=["Date","SecuritiesCode","Section","Target"]+[f"Target{i}" for i in range(seq-1)]).values, valid_xy["Target"].values)  #[valid_xy['Date']==lastday-1]
#custom_loss  'rmse'
model = LGBMRegressor(num_leaves=128,num_iterations = 1000,learning_rate=0.01,objective = 'mse' ,metric='custom', verbose=-1, lambda_l1=1,lambda_l2=1,min_child_weight=30,random_state=random_seed,n_jobs=48 )
model.fit(train_x,train_y, eval_set=watchlist ,eval_metric = custom_lgb_eval, early_stopping_rounds=1000,verbose=20)
print("最优迭代步:",model.best_iteration_, model.best_score_)
score = evaluation_valid(valid_xy,model)
print("valid_score:",score)
joblib.dump(model,f"lgb_model{random_seed}.pkl")   #model=LGBMRegressor()  model=joblib.load(f"lgb_model{random_seed}.pkl")

fea_ = model.feature_importances_
fea_name = train_x.columns.tolist()
fea_name,fea_ = zip(*sorted(zip(fea_name,fea_), key=lambda x:x[1],reverse = True))  #默认,reverse = False升序
print("特征重要程度排序:\n",list(zip(fea_name[:50],fea_[:50])) )
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.barh(fea_name,fea_,height =0.5)
plt.show()


print("总耗时min:",(time.time()-time_now)/60)

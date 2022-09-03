import numpy as np
import pandas as pd
import joblib,re,copy
from lightgbm import LGBMRegressor

train_fin = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv",low_memory=False)  #各个季度的季度情况
train_op = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/options.csv",low_memory=False)
train_sec_pr = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv",low_memory=False)#.rename(columns={'RowId':'DateCode'})
train_pr = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv",low_memory=False)
train_trd = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/trades.csv",low_memory=False).dropna(how='any').reset_index(drop=True)   #前一个交易周的市场总交易情况

stock_list = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv",low_memory=False).rename(columns={'Section/Products':'Section','Close':'Close_MarketCapitalization'})
for f in ["17SectorName","17SectorCode","33SectorName","33SectorCode"]:
    stock_list[f] = stock_list[f].apply(lambda x:np.nan if x=='－' else x.strip())
valid_fin = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/financials.csv",low_memory=False)  #各个季度的季度情况
valid_op = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/options.csv",low_memory=False)
valid_sec_pr = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/secondary_stock_prices.csv",low_memory=False)
valid_pr = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv",low_memory=False)
valid_trd = pd.read_csv("/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/trades.csv",low_memory=False).dropna(how='any').reset_index(drop=True)

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
    train_pr = train_pr.merge(train_trd, how='left',on=['Section','trd_Date']).drop(columns=['StartDate','EndDate'])  #只留下市场各机构销售额之类的

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
    train_fin = pd.concat((train_fin,train_fin1)).reset_index()
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
    train_fin.rename(columns={"Close":"Close_fin","Target":"Target_fin"},inplace=True)
    
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

    train_fin1 = copy.deepcopy(train_fin)
    train_fin1['CurrentPeriodEndDate'] = train_fin1['CurrentPeriodEndDate'].apply(lambda x:month_add(x,1))
    train_fin = pd.concat((train_fin,train_fin1))
    train_fin1['CurrentPeriodEndDate'] = train_fin1['CurrentPeriodEndDate'].apply(lambda x:month_add(x,1))
    train_fin = pd.concat((train_fin,train_fin1)).reset_index()
    train_pr = train_pr.merge(train_fin,how='left',on=['SecuritiesCode','CurrentPeriodEndDate'])
    
    return train_pr,train_fin,train_trd
def add_rank(x,ranks):
    x = x.sort_values(by="Prediction", ascending=False)
    x['Rank'] = range(ranks,ranks+x.shape[0])
    return x
seq = 2
valid_pr,valid_fin,valid_trd = pd.concat((train_pr,valid_pr)).reset_index(drop=True),pd.concat((train_fin,valid_fin)).reset_index(drop=True),pd.concat((train_trd,valid_trd)).reset_index(drop=True)
train_xy,_,_ = process(train_fin,train_op,train_pr,train_sec_pr,train_trd )
df_xy,df_fin,df_trd = process(valid_fin,valid_op,valid_pr,valid_sec_pr,valid_trd )
Drop_fea = [col for col in train_xy.columns if train_xy[col].dtype=='object'] + ["Close","Open","Low","High",'Volume','AdjustmentFactor','SupervisionFlag','ExpectedDividend', 'C*V', 'IssuedShares','trd_Date', 'CurrentPeriodEndDate']

model=LGBMRegressor()
model=joblib.load("/kaggle/input/model0/lgb_model42.pkl")
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

import jpx_tokyo_market_prediction
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

#可能B榜需要不断concat一个df来临时滑窗，但只取最后一个滑窗结果即可
for (prices,options,financials,trades,secondary_prices,sample_prediction) in iter_test:
    #通过df_xy不断粘上新的df
    df,_,_ = process(financials,options,prices,secondary_prices,trades )
    df["Target"] = 0
    df_xy = pd.concat((df_xy,df))
    df = df_xy[df_xy["Date"]<=df["Date"].max()].sort_values("Date").drop_duplicates(["Date","SecuritiesCode"],keep='last').reset_index(drop=True)
    #若需要做因子，则用该df做，需要加速，则尝试递推公式
    df = df.groupby('SecuritiesCode').apply(lambda x:MACD_add(x)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:K_add(x,1)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:K_add(x,3)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:K_add(x,5)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:K_add(x,9)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:other_add(x)).reset_index(drop=True)
    df = df.groupby('SecuritiesCode').apply(lambda x:KDJ_add(x)).reset_index(drop=True)
    df = df.drop(columns=Drop_fea)
    fea_L = df.shape[1]-3
    #做完因子再取出最后需要的序列df，减少get_window计算时间
    df = df.groupby('SecuritiesCode').apply(lambda x:get_window(x.iloc[-seq:].sort_values('Date'),seq)).reset_index(drop=True)
    data = sample_prediction[["Date","SecuritiesCode"]]
    data["Date"] = data["Date"].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    data["SecuritiesCode"] = data["SecuritiesCode"].astype("int")
    data = data.merge(df,how='left',on=["Date","SecuritiesCode"])
    data["Prediction"] = model.predict( data.drop(columns=["Date","SecuritiesCode","Target"]+[f"Target{i}" for i in range(seq-1)]).values,num_iteration=model.best_iteration_ )
    data = add_rank(data,0).reset_index(drop=True)[["SecuritiesCode","Rank"]]
    sample_prediction = sample_prediction.drop(columns="Rank").merge(data,how='left',on=["SecuritiesCode"])
    env.predict(sample_prediction)
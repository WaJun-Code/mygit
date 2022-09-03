import numpy as np
import pandas as pd
import warnings,random,os,tqdm,re
import torch
import torch.nn as nn
import torch.nn.functional as F
import time,copy

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process(train_fin,train_op,train_pr,train_sec_pr,train_trd ):
    for f in [train_fin,train_op,train_pr,train_sec_pr,train_trd ]:
        f['Date'] = f['Date'].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    train_pr['C*V'] = train_pr['Volume']*train_pr['Close']
    #先粘上股票信息，带入了市场部门名称，但需要注意stock表里有生效日期，如何处理？
    train_pr = train_pr.merge(stock_list[['SecuritiesCode','IssuedShares','Section','33SectorCode','17SectorCode']],how='left',on='SecuritiesCode')
    #换手率直接加，成交量/发行股份
    train_pr['Turnover'] = train_pr['Volume']/train_pr['IssuedShares']
    return train_pr,train_fin,train_trd

seq = 5+2
valid_pr,valid_fin,valid_trd = pd.concat((train_pr,valid_pr)).reset_index(drop=True),pd.concat((train_fin,valid_fin)).reset_index(drop=True),pd.concat((train_trd,valid_trd)).reset_index(drop=True)
train_xy,_,_ = process(train_fin,train_op,train_pr,train_sec_pr,train_trd )
df_xy,df_fin,df_trd = process(valid_fin,valid_op,valid_pr,valid_sec_pr,valid_trd )
Drop_fea = [col for col in df_xy.columns if df_xy[col].dtype=='object'] + ["Open","Low","High",'Volume','AdjustmentFactor','SupervisionFlag','ExpectedDividend']
df_xy = df_xy.drop(columns=Drop_fea)
fea_L = df_xy.shape[1]-3
fea = ["Close"]+df_xy.drop(columns=["Close"]).columns.tolist()  #Close放第一个
df_xy = df_xy[fea]
for col in df_xy.columns:  #归一化方法也有待斟酌
    maxmin_Close = train_xy[col].quantile([0.01,0.99])
    if col in {'Date','SecuritiesCode',"Target"} or df_xy[col].nunique()<10 or maxmin_Close[0.99]==maxmin_Close[0.01]:continue
    df_xy[col] = (df_xy[col]-maxmin_Close[0.01])/(maxmin_Close[0.99]-maxmin_Close[0.01])
df_xy = df_xy.fillna(0)

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        #train_xy.shape[1]-3
        self.lstm0 = nn.LSTM(1,128)
        self.lstm1 = nn.LSTM(128,1)
    def forward(self,X,online=False):   #输入一个【bz，seq+2，dim】,可以在这多录入2个，形成seq+2，好加loss
        X = X.permute(1,0,2)
        X,C1,C2 = X[:seq-2,:,:],X[1:-1,:,:],X[2:,:,:]
        out = self.lstm0(X)[0]
        out1 = self.lstm1(out)[0]
        #out = torch.cat((X[1:,:,:],out1[-1,:,:]),dim=0)
        #重新喂入out1还是X？
        out = self.lstm0(out1)[0]
        out = self.lstm1(out)[0]
        if online==False:
            loss1,loss2 = nn.MSELoss()(out1, C1 ),nn.MSELoss()(out, C2 )   #可以考虑仅用最后一个Close、Target来做loss
        out = (out-out1)/out1
        out = out.permute(1,0,2).flatten(1)
        if online:return out
        else:return out,loss1+loss2
LSTM = LSTM().to(device)
LSTM.load_state_dict(torch.load("/kaggle/input/model0/LSTM42.pth", map_location=device))
LSTM.eval()

def add_rank(x,ranks):
    x = x.sort_values(by="Prediction", ascending=False)
    x['Rank'] = range(ranks,ranks+x.shape[0])
    return x
def get_res(model0,batch,ranks):  #L为P_L或N_L，P_L的ranks=0，N_L的ranks=len(P_L)+1
    y_pred0,Target = [],[]
    seq0 = seq-2
    with torch.no_grad():
        y_d = [i for i in range(1,seq0)]+[seq0]*(batch.shape[0]-2*seq0+2)+[seq0-i for i in range(1,seq0)]
        y_pred00 = np.zeros([batch.shape[0],])
        X,T = batch[[f"fea{i}" for i in range(seq0*fea_L )]].values.reshape(-1,seq0,fea_L ).tolist(), batch[[f"Target{i}" for i in range(seq0-1)]+["Target"]].values.reshape(-1,seq0).tolist()
        X,T = torch.tensor(X,dtype=torch.float32).to(device), torch.tensor(T,dtype=torch.float32).to(device)
        X0,X1 = X[seq0-1:,:,:1],X[:,:,1:]
        out0 = model0(X0,True)  #[bz,seq-2]放入y_
        out0 = out0.detach().cpu().numpy()
        y_pred0 += out0[:,-1].tolist()
        # for j in range(batch.shape[0]-seq0):y_pred00[j: seq0+j] += out0[j,:]
        # y_pred00 = y_pred00/y_d
        # y_pred0 += y_pred00.tolist()
    valid_0 = batch["Date"].copy().to_frame()
    valid_0["SecuritiesCode"] = batch["SecuritiesCode"].copy()
    #valid_0["Prediction"] = y_pred0/np.array(y_d)
    valid_0["Prediction"] = y_pred0
    valid_0["Target"] = batch["Target"].copy()
    valid_0 = add_rank(valid_0,ranks).reset_index(drop=True)
    return valid_0
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
    df = df.drop(columns=Drop_fea)[fea]
    for col in df.columns:  #归一化方法也有待斟酌
        maxmin_Close = train_xy[col].quantile([0.01,0.99])
        if col in {'Date','SecuritiesCode',"Target"} or df[col].nunique()<10 or maxmin_Close[0.99]==maxmin_Close[0.01]:continue
        df[col] = (df[col]-maxmin_Close[0.01])/(maxmin_Close[0.99]-maxmin_Close[0.01])
    df = df.fillna(0)
    df_xy = pd.concat((df_xy,df))
    df = df_xy[df_xy["Date"]<=df["Date"].max()].sort_values("Date").drop_duplicates(["Date","SecuritiesCode"],keep='last').reset_index(drop=True)   #若需要做因子，则用该df做
    #做完因子再取出最后需要的序列df，减少get_window计算时间
    df = df.groupby('SecuritiesCode').apply(lambda x:get_window(x.iloc[-seq+2:].sort_values('Date'),seq)).reset_index(drop=True)
    data = sample_prediction[["Date","SecuritiesCode"]]
    data["Date"] = data["Date"].apply(lambda x :int(re.sub(r'[^a-zA-Z0-9]','',x)) )
    data["SecuritiesCode"] = data["SecuritiesCode"].astype("int")
    data = data.merge(df,how='left',on=["Date","SecuritiesCode"])
    data = get_res(LSTM,data,0)[["SecuritiesCode","Rank"]]
    sample_prediction = sample_prediction.drop(columns="Rank").merge(data,how='left',on=["SecuritiesCode"])
    env.predict(sample_prediction)

import numpy as np
import pandas as pd
'''
train_xy.Date = pd.to_datetime(train_xy.Date)
train_xy['Date'] = train_xy['Date'].dt.strftime("%Y%m%d").astype(int)
'''
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import warnings,random,os,tqdm,re
import time,copy
warnings.filterwarnings("ignore")

def MACD_add(data_fct,N = [12,26], M = 9):  #注意使用前需要groupby一下
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 计算中间变量
    EMA1 = data['Close'].ewm(span = N[0], adjust = False).mean()
    EMA2 = data['Close'].ewm(span = N[1], adjust = False).mean()
    DIFF = EMA1 - EMA2
    DEA = DIFF.ewm(span = M, adjust = False).mean()
    
    # 计算因子
    MACD = 2*(DIFF - DEA)
    MACD.rename('MACD',inplace = True)
    DIFF.rename('DIFF',inplace = True)
    DEA.rename('DEA',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(DIFF,left_index = True,right_index = True, how = 'left')
    output = output.merge(DEA,left_index = True,right_index = True, how = 'left')
    output = output.merge(MACD,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def ACD_add(data_fct,N = 20):
    # DIF计算函数
    def _DIF_f(series):
        if series['Close'] > series['PreClose']:
            return(series['Close'] - min(series['Low'],series['PreClose']))
        elif series['Close'] < series['PreClose']:
            return(series['Close'] - max(series['High'],series['PreClose']))
        else:
            return(0)

    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    data['PreClose'] = data['Close'].shift(1)    # 添加昨日收盘价
    data['DIF'] = data.apply(_DIF_f, axis = 1)    # 添加中间变量DIF计算
    ACD = data['DIF'].rolling(N).sum()    # 添加因子计算
    ACD.rename('ACD',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(ACD,left_index = True,right_index = True)
    # 返回增加因子的数据框
    return output

#写一个按周或者天K线图对应特征的
def K_add(df,N):
    tmp = df['Close'].to_frame()
    tmp['Open'] = df['Open'].shift(N-1)   #shift正方向为往前移
    #tmp = tmp.fillna(method='bfill')   #用第一个open填开头N个缺失地方
    C_O = (tmp['Close'] - tmp['Open'] ).rename(f'C_O{N}')
    M_L = (tmp.min(axis=1) - df['Low'].rolling(N).min() ).rename(f'M_L{N}')
    H_M = (df['High'].rolling(N).max() - tmp.max(axis=1) ).rename(f'H_M{N}')
    
    df = df.merge(C_O,left_index = True,right_index = True,how = 'left')
    df = df.merge(M_L,left_index = True,right_index = True,how = 'left')
    df = df.merge(H_M,left_index = True,right_index = True,how = 'left')
    return df

def other_add(data_fct,N = 10):
    data = copy.deepcopy(data_fct)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 计算中间变量
    C_Diff = data['Close'] - data['Close'].shift(1)
    U = C_Diff.apply(lambda series: series if series > 0 else 0)
    D = C_Diff.apply(lambda series: -series if series < 0 else 0)
    # 计算中间变量的指数移动平均
    EMA_U = U.ewm(span = N, adjust = False).mean()
    EMA_D = D.ewm(span = N, adjust = False).mean()
    #计算相对强弱指数
    RSI = EMA_U/(EMA_U+EMA_D) * 100
    RSI.rename(f'RSI{N}',inplace = True)
    output = output.merge(RSI,left_index = True,right_index = True,how = 'left')
    #计算过去5日对应的乖离率
    Bias = data['Close'].rolling(N).mean()
    Bias = (data['Close'] - Bias)/Bias
    Bias.rename(f'Bias{N}',inplace = True)
    output = output.merge(Bias,left_index = True,right_index = True,how = 'left')
    
    #量比，此处注意需要volume列
    Bias = data['Volume'].rolling(N).mean()
    Bias = (data['Volume'] - Bias)/Bias
    Bias.rename(f'Volume_ratio{N}',inplace = True)
    output = output.merge(Bias,left_index = True,right_index = True,how = 'left')
    return output

#计算KDJ
def KDJ_add(data_fct,N = 9):
    indexs = data_fct.index.tolist()
    def _KDJ_f(series):  #注意series的下标，不能改变原df下标
        kdj = series.copy()
        for i in range(len(series)):
            if i == 0:
                kdj[indexs[i]] = 50
            else:
                kdj[indexs[i]] = 2/3 * kdj[indexs[i-1]] + 1/3 * series[indexs[i]]
        return(kdj)
    
    # 深拷贝制作数据副本
    data = copy.deepcopy(data_fct)
    
    # 添加中间变量
    Lowest = data['Low'].rolling(N).min()
    Highest = data['High'].rolling(N).max()
    RSV = (data['Close'] - Lowest)/(Highest - Lowest)*100
    RSV = RSV.fillna(method='ffill').fillna(50)  #防止出现nan传递现象，出现nan则从前一个开始递推
    
    # 添加KDJ
    KDJ_K = _KDJ_f(RSV)
    KDJ_K.rename(f'KDJ_K{N}',inplace = True)
    KDJ_D = _KDJ_f(KDJ_K)
    KDJ_D.rename(f'KDJ_D{N}',inplace = True)
    KDJ_J = 3 * KDJ_D - 2 * KDJ_K
    KDJ_J.rename(f'KDJ_J{N}',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(KDJ_K,left_index = True,right_index = True,how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(KDJ_D,left_index = True,right_index = True,how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(KDJ_J,left_index = True,right_index = True,how = 'left')
    
    # 返回增加因子的数据框
    return output


#计算ASI
def ASI_add(data_fct, N = 20):
    # 振动升降指标计算函数
    def _SI_f(series):
        A = abs(series['High'] - series['PreClose'])
        B = abs(series['Low'] - series['PreClose'])
        C = abs(series['High'] - series['PreLow'])
        D = abs(series['PreClose'] - series['PreOpen'])
        E = series['Close'] - series['PreClose']
        F = series['Close'] - series['Open']
        G = series['PreClose'] - series['PreOpen']
        X = E+0.5*F+G
        K = max(A,B)
        if A > B and A > C:
            R = A + 0.5*B + 0.25*D
        elif B > A and B > C:
            R = B + 0.5*A + 0.25*D
        else:
            R = C + 0.25*D
        SI = 16*X/R*K
        return(SI)
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    data['PreClose'] = data['Close'].shift(1)    # 添加昨日收盘价
    data['PreLow'] = data['Low'].shift(1)    # 添加昨日最低价
    data['PreOpen'] = data['Open'].shift(1)    # 添加昨日开盘价
    data['SI'] = data.apply(_SI_f, axis = 1)    # 获取每日的振动升降指标
    ASI = data['SI'].rolling(N).sum()    # 获取累计的振动升降指标
    ASI.rename('ASI',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(ASI,left_index = True,right_index = True)
    output = output.merge(data['SI'],left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def AR_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    data['H_O'] = data['High'] - data['Open']    # 添加最高价减开盘价的价差
    data['O_L'] = data['Open'] - data['Low']    # 添加开盘价减最低价的价差
    data['H_O_N'] = data['H_O'].rolling(N).sum()    # 添加移动求和
    data['O_L_N'] = data['O_L'].rolling(N).sum()
    # 添加AR计算
    AR = data['H_O_N']/data['O_L_N'] * 100
    AR.rename('AR',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(AR,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def BR_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    PreClose = data['Close'].shift(1)    # 添加昨日收盘价
    H_PreC = data['High'] - PreClose    # 添加最高价减开盘价的价差
    PreC_L = PreClose - data['Low']    # 添加开盘价减最低价的价差
    # 添加移动求和
    H_PreC_N = H_PreC.rolling(N).sum()
    PreC_L_N = PreC_L.rolling(N).sum()
    
    # 添加BR计算
    BR = H_PreC_N/PreC_L_N * 100
    BR.rename('BR',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BR,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def High52_add(data_fct,N = 250):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    High52 = data['Close']/data['High'].rolling(N).max()    # 添加过去52周中的最高价
    High52.rename('High52',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(High52,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def PVT_add(data_fct,M = 6):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    PreClose = data['Close'].shift(1)    # 添加昨日收盘价
    # 计算PVT(暂不计算累计)
    PVT = (data['Close']/PreClose-1 ) * data['Volume']
    PVT.rename('PVT',inplace = True)
    PVT6 = PVT.rolling(M).mean()
    PVT6.rename('PVT6',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(PVT,left_index = True,right_index = True, how = 'left')
    output = output.merge(PVT6,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def OBV_add(data_fct,N = [6,20]):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    C_PreC = data['Close'] - data['Close'].shift(1)
    Sign = np.sign(C_PreC)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    #计算因子
    OBV = (data['Volume'] * Sign).cumsum()
    OBV.rename('OBV',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(OBV,left_index = True,right_index = True, how = 'left')
    for n in N:
        MAOBV = OBV.rolling(n).mean()
        MAOBV.rename('OBV'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(MAOBV,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def CVI_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    H_L = data['High'] - data['Low']    # 添加最高价和最低价的价差
    EMA = H_L.ewm(span = N, adjust = False).mean()    # 添加中间变量计算
    # 计算CVI
    CVI = (EMA-EMA.shift(N))/EMA *100
    CVI.rename('CVI',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(CVI,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def ADTM_add(data_fct, N = 23, M = 8):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    data['PreOpen'] = data['Open'].shift(1)    # 添加昨日收盘价
    
    # 添加中间变量计算
    data['O_PreO'] = data['Open'] - data['PreOpen']
    data['H_O'] = data['High']-data['Open']
    data['O_L'] = data['Open']-data['Low']
    data['Temp_Max1'] = data[['H_O','O_PreO']].max(axis = 1)
    data['Temp_Max2'] = data[['O_L','O_PreO']].max(axis = 1)
    # 计算DTM
    data['DTM'] = data.apply(lambda series: series['Temp_Max1'] if series['O_PreO'] >0 else 0,axis = 1)
    # 计算DBM
    data['DBM'] = data.apply(lambda series: series['Temp_Max2'] if series['O_PreO'] <0 else 0,axis = 1)
    # 计算STM
    data['STM'] = data['DTM'].rolling(N).sum()
    # 计算SBM
    data['SBM'] = data['DBM'].rolling(N).sum()
    #添加中间变量计算
    data['STM_SBM'] = data['STM'] - data['SBM']
    data['T_B/T'] = data['STM_SBM']/data['STM']
    data['T_B/B'] = data['STM_SBM']/data['SBM']
    # 计算ADTM
    ADTM = data.apply(lambda series: series['T_B/T'] if series['STM_SBM'] > 0 else series['T_B/B'],axis = 1)
    ADTM.rename('ADTM',inplace = True)
    # 计算MAADTM
    MAADTM = ADTM.rolling(M).mean()
    MAADTM.rename('MAADTM',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    output = output.merge(ADTM,left_index = True,right_index = True, how = 'left')
    output = output.merge(MAADTM,left_index = True,right_index = True, how = 'left')
    output = output.merge(data['SBM'],left_index = True,right_index = True, how = 'left')
    output = output.merge(data['STM'],left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def CCI_add(data_fct,N = [5,10,20,88]):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 计算移动平均BIAS
    for n in N:
        # 添加中间变量
        TP = (data['High'] + data['Low'] + data['Close'])/3
        MA = data['Close'].rolling(n).mean()
        TP_MA = TP - MA
        MD = TP_MA.rolling(n).mean()

        # 添加BollUp计算
        CCI = (TP-MA)/(MD*0.015)
        CCI.rename('CCI' + str(n),inplace = True)
    
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(CCI,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def ATR_add(data_fct,N = [6,14]):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    
    # 添加昨日收盘价
    PreClose = data['Close'].shift(1)
    H_PreC = abs(data['High'] - PreClose)
    L_PreC = abs(data['Low'] - PreClose)
    H_L = data['High'] - data['Low']
    # 添加中间变量计算
    TR = (H_L + H_PreC + L_PreC)/3
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 计算移动平均ATR
    for n in N:
        ATR = TR.rolling(n).mean()
        ATR.rename('ATR' + str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(ATR,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def CR_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    Mid = (data['High'] + data['Low'])/2    # 添加中间价
    PreMid = Mid.shift(1)    # 添加昨日中间价
    Up = data['High'] - PreMid    # 添加上升值
    Down = PreMid - data['Low']    # 添加下跌值
    Up_N = Up.rolling(N).sum()    # 添加移动求和
    Down_N = Down.rolling(N).sum()
    
    # 添加CR计算
    CR = Up_N/Down_N * 100
    CR.rename('CR',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(CR,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def MassIndex_add(data_fct,N1 = 9, N2 = 25):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    H_L = data['High'] - data['Low']    # 添加最高价和最低价的价差
    # 添加中间变量计算
    EMA1 = H_L.ewm(span = N1, adjust = False).mean()
    EMA2 = EMA1.ewm(span = N1, adjust = False).mean()
    
    DI = EMA1/EMA2
    # 计算MassIndex
    MassIndex = DI.rolling(N2).sum()
    MassIndex.rename('MassIndex',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(MassIndex,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def Boll_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    MA = data['Close'].rolling(N).mean()    # 添加中间变量
    Std = data['Close'].rolling(N).std()
    BollUp = MA + 2*Std     # 添加BollUp计算
    BollUp.rename('BollUp',inplace = True)
    
    # 添加BollDown计算
    BollDown = MA - 2*Std 
    BollDown.rename('BollDown',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BollUp,left_index = True,right_index = True, how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BollDown,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def Elder_add(data_fct,N = 13):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    EMA_C = data['Close'].ewm(span = N, adjust = False).mean()    # 添加中间变量
    BullPower = data['High'] - EMA_C    # 计算多头力道 
    BullPower.rename('BullPower',inplace = True)    # 计算空头力道 
    BearPower = data['Low'] - EMA_C
    BearPower.rename('BearPower',inplace = True)
    # 计算艾达透视指标
    Elder = (BullPower-BearPower)/data['Close']
    Elder.rename('Elder',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BullPower,left_index = True,right_index = True, how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BearPower,left_index = True,right_index = True, how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(Elder,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def BBI_add(data_fct,N = [3,6,12,24]):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    
    Temp = pd.DataFrame(index = data.index)    # 用于暂存中间变量
    # 添加中间变量
    for n in N:
        MA = data['Close'].rolling(n).mean()
        MA.rename('MA' + str(n),inplace = True)
        Temp = Temp.merge(MA,left_index = True,right_index = True)
    Temp.dropna(inplace = True)
    
    # 添加BBI计算
    names = ['MA'+ str(element) for element in N]
    BBI = np.mean(Temp[names],axis = 1)
    BBI.rename('BBI',inplace = True)
    # 添加BBIC计算
    BBIC = BBI/data['Close']
    BBIC.rename('BBIC',inplace = True)
    
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BBI,left_index = True,right_index = True,how = 'left')
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(BBIC,left_index = True,right_index = True,how = 'left')
    # 返回增加因子的数据框
    return output

def MA_add(data_fct,N = [5,10,20,60,120]):
    output = copy.deepcopy(data_fct)    # 制作用于最终输出的数据框copy
    # 计算移动平均MA
    for n in N:
        # 添加MA
        MA = output['Close'].rolling(n).mean()
        MAname = 'MA' + str(n)
        MA.rename(MAname,inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(MA,left_index = True,right_index = True,how = 'left')
    # 返回增加因子的数据框
    return output

def EMA_add(data_fct,N = [5,10,12,20,26,60,120]):
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 对参数列表中的参数进行循环
    for n in N:
        # 添加EMA
        EMA = output['Close'].ewm(span = n, adjust = False).mean()
        EMAname = 'EMA' + str(n)
        EMA.rename(EMAname,inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(EMA, left_index = True, right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def TEMA_add(data_fct,N = [5,10]):
    data = copy.deepcopy(data_fct)    # 深拷贝制作数据副本
    output = copy.deepcopy(data_fct)    # 制作用于最终输出的数据框copy
    
    for n in N:
        # 添加TEMA的中间变量
        TEMA1 = data['Close'].ewm(span = n, adjust = False).mean()
        TEMA2 = TEMA1.ewm(span = n, adjust = False).mean()
        TEMA3 = TEMA2.ewm(span = n, adjust = False).mean()
        # 计算三重指数移动平均
        TEMA = (TEMA1+TEMA2+TEMA3)/3
        TEMAname = 'TEMA' + str(n)
        TEMA.rename(TEMAname,inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(TEMA,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def DBCD_add(data_fct,N = 5,M = 16,T = 17):
    data = copy.deepcopy(data_fct)
    # 计算中间变量
    MA = data['Close'].rolling(N).mean()
    BIAS = (data['Close']/MA-1)*100
    DIF = BIAS - BIAS.shift(M)
    DIF.dropna(inplace = True)
    # 计算异同离差乖离率
    DBCD = DIF.ewm(span = T, adjust = False).mean()
    DBCD.rename('DBCD',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(DBCD,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def MFI_add(data_fct,N = 14):
    data = copy.deepcopy(data_fct)
    # 计算中间变量
    TP = (data['Close'] + data['High'] + data['Low'])/3
    MF = TP*data['Volume']
    MF_Diff = MF - MF.shift(1)
    MF_Diff.dropna(inplace = True)
    PMF = MF_Diff.apply(lambda series: series if  series > 0 else 0)
    NMF = MF_Diff.apply(lambda series: -series if  series < 0 else 0)
    MR = PMF.rolling(N).sum()/NMF.rolling(N).sum()
    
    # 计算资金流量指标
    MFI = 100 *(MR/(1+MR))
    MFI.rename('MFI',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(MFI,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def MTM_add(data_fct,N1 = 10,N2 = 10):
    data = copy.deepcopy(data_fct)
    
    # 计算MTM变量
    MTM = data['Close'] - data['Close'].shift(N1)
    MTM.rename('MTM',inplace = True)
    
    MTMMA = MTM.rolling(N2).mean()
    MTMMA.rename('MTMMA'+str(N2),inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(MTM,left_index = True,right_index = True,how = 'left')
    output = output.merge(MTMMA,left_index = True,right_index = True,how = 'left')
    # 返回增加因子的数据框
    return output

def Ulcer_add(data_fct,N = [5,10]):
    data = copy.deepcopy(data_fct)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
              
    # 计算MTM变量
    for n in N:
        C_Max = data['Close'].rolling(n).max()
        Ulcer = (data['Close'] - C_Max)/C_Max
        Ulcer.rename('Ulcer' + str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(Ulcer,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def DHILO_add(data_fct,N = 60):
    data = copy.deepcopy(data_fct)
    # 中间变量计算
    lgH_lgL = np.log(data['High']) - np.log(data['Low'])
    # 计算波幅中位数
    DHILO = lgH_lgL.rolling(N).median()
    DHILO.rename('DHILO',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(DHILO,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def EMV_add(data_fct,N = [6,14]):
    data = copy.deepcopy(data_fct)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 中间变量计算
    H_L_Avg = data[['High','Low']].mean(axis = 1)
    H_L = data['High']-data['Low']
    Temp = (H_L_Avg-H_L_Avg.shift(1))*H_L/data['Volume']
    Temp.dropna(inplace = True)
    
    # 对于每个可能的参数进行计算
    for n in N:
        # 计算指数移动平均得到简易波动指标
        EMV = Temp.ewm(span = n, adjust = False).mean()
        EMV.rename('EMV'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(EMV,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def RVI_add(data_fct,N = 10):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    data['Std'] = data['Close'].rolling(N).std()
    data['C_PreC'] = data['Close'] - data['Close'].shift(1)
    USD = data.apply(lambda series: series['Std'] if series['C_PreC'] > 0 else 0 ,axis = 1)
    DSD = data.apply(lambda series: series['Std'] if series['C_PreC'] < 0 else 0 ,axis = 1)
    
    #最终因子计算
    UpRVI = USD.ewm(span = 2*N-1, adjust = False).mean()
    UpRVI.rename('UpRVI',inplace = True)
    DownRVI = DSD.ewm(span = 2*N-1, adjust = False).mean()
    DownRVI.rename('DownRVI',inplace = True)
    RVI = 100 * UpRVI / (UpRVI + DownRVI)
    RVI.rename('RVI',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 中间变量计算

    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(UpRVI,left_index = True,right_index = True,how = 'left')
    output = output.merge(DownRVI,left_index = True,right_index = True,how = 'left')
    output = output.merge(RVI,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def ChaikinOscillator_add(data_fct,N = [3,10]):
    data = copy.deepcopy(data_fct)
    # 计算中间变量
    M = ((data['Close'] - data['Low']) - (data['High'] - data['Close']))/(data['High'] - data['Low'])*data['Volume']
    ADL = M + M.shift(1)
    ADL.dropna(inplace = True)
    
    #最终因子计算
    print(N[0])
    print(N[1])
    CO = ADL.ewm(span = N[0], adjust = False).mean() - ADL.ewm(span = N[1], adjust = False).mean()
    CO.rename('ChaikinOscillator',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)

    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(CO,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def ChaikinVolatility_add(data_fct,N = 10):
    data = copy.deepcopy(data_fct)
    # 计算中间变量
    HLEMA = (data['High'] - data['Low']).ewm(span = N, adjust = False).mean()
    
    # 计算最终的因子
    CV = 100 * (HLEMA/HLEMA.shift(10) - 1)
    CV.rename('ChaikinVolatility',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)

    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(CV,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def UOS_add(data_fct,M =5, N = 10, O =20):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    data['PreClose'] = data['Close'].shift(1)
    TH = data[['High','PreClose']].max(axis = 1)
    TL = data[['Low','PreClose']].max(axis = 1)
    TR = TH - TL
    XR = data['Close'] - TL
    XRM = XR.rolling(M).sum()/TR.rolling(M).sum()
    XRN = XR.rolling(N).sum()/TR.rolling(N).sum()
    XRO = XR.rolling(O).sum()/TR.rolling(O).sum()
    # 计算最终的因子
    UOS = 100 * (XRM*N*O + XRN*M*O + XRO*M*N) / (M*N + M*O+ N*O)
    UOS.rename('UOS',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)

    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(UOS,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def DMI_add(data_fct, N = 14):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    H_PreH = data['High'] - data['High'].shift(1)
    PreL_L = data['Low'].shift(1) - data['Low']
    PlusDM = H_PreH.apply(lambda series : series if series > 0 else 0)
    PlusDM.dropna(inplace = True)
    MinusDM = PreL_L.apply(lambda series : series if series > 0 else 0)
    MinusDM.dropna(inplace = True)
    data['H_L'] = data['High'] - data['Low']
    data['H_PreC'] = data['High'] - data['Close'].shift(1)
    data['L_PreC'] = data['Low'] - data['Close'].shift(1)
    TR = data[['H_L','H_PreC','L_PreC']].max(axis = 1)
    PlusDI = PlusDM.ewm(span = N, adjust = False).mean()/TR.ewm(span = N, adjust = False).mean() *100
    MinusDI = MinusDM.ewm(span = N, adjust = False).mean()/TR.ewm(span = N, adjust = False).mean() *100
    DX = abs(PlusDI - MinusDI)/abs(PlusDI + MinusDI) *100

    # 计算最终的因子
    PlusDI.rename('PlusDI',inplace = True)
    MinusDI.rename('MinusDI',inplace = True)
    ADX = DX.ewm(span = N, adjust = False).mean()
    ADX.rename('ADX',inplace = True)
    ADXR = (ADX + ADX.shift(N))/2
    ADXR.rename('ADXR',inplace = True)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)

    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(PlusDI,left_index = True,right_index = True,how = 'left')
    output = output.merge(MinusDI,left_index = True,right_index = True,how = 'left')
    output = output.merge(ADX,left_index = True,right_index = True,how = 'left')
    output = output.merge(ADXR,left_index = True,right_index = True,how = 'left')

    # 返回增加因子的数据框
    return output

def ARBR_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)
    # AR中间变量
    H_O = data['High'] - data['Open']
    O_L = data['Open'] - data['Low']
    H_O_N = H_O.rolling(N).sum()
    O_L_N = O_L.rolling(N).sum()
    
    # 添加AR计算
    AR = H_O_N/O_L_N * 100
    
    # BR中间变量
    PreClose = data['Close'].shift(1)
    H_PreC = data['High'] - PreClose
    PreC_L = PreClose - data['Low']
    H_PreC_N = H_PreC.rolling(N).sum()
    PreC_L_N = PreC_L.rolling(N).sum()
    
    # 添加BR计算
    BR = H_PreC_N/PreC_L_N * 100
    
    # 计算ARBR
    ARBR = AR - BR
    ARBR.rename('ARBR',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(ARBR,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def PSY_add(data_fct,N = 12, M = 6):
    data = copy.deepcopy(data_fct)
    
    # 中间变量计算
    Diff = data['Close'] - data['Close'].shift(1)
    Label = Diff.apply(lambda series: 1 if series > 0 else 0)

    #计算PSY
    PSY = Label.rolling(N).mean()
    PSY.rename('PSY',inplace = True)
    MAPSY = PSY.rolling(M).mean()
    MAPSY.rename('MAPSY',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(PSY,left_index = True,right_index = True, how = 'left')
    output = output.merge(MAPSY,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def WVAD_add(data_fct,N = 24):
    data = copy.deepcopy(data_fct)
    
    # 计算因子
    WVAD = ((data['Close'] - data['Open'])/(data['High'] - data['Low']) * data['Volume']).rolling(N).sum()
    WVAD.rename('WVAD',inplace = True)
    MAWVAD = WVAD.rolling(N).mean()
    MAWVAD.rename('MAWVAD',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(WVAD,left_index = True,right_index = True, how = 'left')
    output = output.merge(MAWVAD,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def ROC_add(data_fct,N = [6,20]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        ROC = (data['Close']/data['Close'].shift(n) - 1)*100
        ROC.rename('ROC'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(ROC,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def ARC_add(data_fct,N = 50):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    RC = data['Close']/data['Close'].shift(N)
    # 计算ARC
    ARC = RC.ewm(alpha=1/N, adjust=False).mean()
    ARC.rename('ARC',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(ARC,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def APBMA_add(data_fct,N = 5):
    data = copy.deepcopy(data_fct)
    
    # 计算APBMA
    APBMA = (abs(data['Close'] - data['Close'].rolling(N).mean())).rolling(N).mean()
    APBMA.rename('APBMA',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(APBMA,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def MA10Close_add(data_fct,N = 10):
    data = copy.deepcopy(data_fct)
    
    # 计算MA10Close
    MA10Close = data['Close'].rolling(N).mean()/data['Close']
    MA10Close.rename('MA10Close',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(MA10Close,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def SRMI_add(data_fct,N = 10):
    data = copy.deepcopy(data_fct)
    
    
    # 计算中间变量
    data['PreNC'] = data['Close'].shift(N)
    C_C_N = data['Close'] - data['PreNC']
    C_C_N_Max = data[['Close','PreNC']].max(axis = 1)
    
    # 计算因子
    SRMI = C_C_N /C_C_N_Max
    SRMI.rename('SRMI',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(SRMI,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def CMO_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)
    
    
    # 计算中间变量
    C_PreC = data['Close'] - data['Close'].shift(1)
    CZ1 = C_PreC.apply(lambda series: series if series > 0 else 0)
    CZ2 = C_PreC.apply(lambda series: -series if series < 0 else 0)
    ChandeSU = CZ1.rolling(N).sum()
    ChandeSD = CZ2.rolling(N).sum()
    
    # 计算因子
    CMO = (ChandeSU - ChandeSD)/(ChandeSU + ChandeSD) * 100
    CMO.rename('CMO',inplace = True)
    ChandeSU.rename('ChandeSU',inplace = True)
    ChandeSD.rename('ChandeSD',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(CMO,left_index = True,right_index = True, how = 'left')
    output = output.merge(ChandeSU,left_index = True,right_index = True, how = 'left')
    output = output.merge(ChandeSD,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def REVS_add(data_fct,N = [5,10,20,60,120,250]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        REVS = data['Close']/data['Close'].shift(n)
        REVS.rename('REVS'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(REVS,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def REVS5M_add(data_fct,N = [20, 60]):
    data = copy.deepcopy(data_fct)
    
    REVS5 = data['Close']/data['Close'].shift(5)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        REVS5M = REVS5 - data['Close']/data['Close'].shift(n)
        REVS5M.rename('REVS5M'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(REVS5M,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def FiftyTwoWeekHigh_add(data_fct,N = 250):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    H_Year = data['Close'].rolling(N).max()
    L_Year = data['Close'].rolling(N).min()
    
    #计算因子
    FiftyTwoWeekHigh = (data['Close'] - L_Year)/(H_Year - L_Year)
    FiftyTwoWeekHigh.rename('FiftyTwoWeekHigh',inplace = True)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(FiftyTwoWeekHigh,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def Price1M_add(data_fct, N = [20,60,250]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        Mean = data['Close'].rolling(n).mean()
        Price = data['Close']/Mean - 1
        if n ==20:
            Price.rename('Price1M',inplace = True)
        if n ==60:
            Price.rename('Price3M',inplace = True) 
        if n == 250:
            Price.rename('Price1Y',inplace = True) 
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(Price,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def RC_add(data_fct,N = [12,24]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        # 计算因子
        RC = data['Close']/data['Close'].shift(n)
        RC.rename('RC'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(RC,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def DDI_add(data_fct,N = 13):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 计算中间变量
    data['HL_PreHL'] = data['High'] + data['Low'] - data['High'].shift(1) - data['Low'].shift(1)
    data['H_PreH'] = abs(data['High'] - data['High'].shift(1))
    data['L_PreL'] = abs(data['Low'] - data['Low'].shift(1))
    data['HL_Max'] = data[['H_PreH','L_PreL']].max(axis = 1)
    DMZ = data.apply(lambda series: series['HL_Max'] if series['HL_PreHL'] > 0 else 0,axis = 1)
    DMF = data.apply(lambda series: series['HL_Max'] if series['HL_PreHL'] < 0 else 0,axis = 1)
    DMZ_Sum = DMZ.rolling(N).sum()
    DMF_Sum = DMF.rolling(N).sum()
    
    # 计算因子
    DIZ = DMZ_Sum/(DMZ_Sum + DMF_Sum)
    DIZ.rename('DIZ',inplace = True)
    DIF = DMF_Sum/(DMZ_Sum + DMF_Sum)
    DIF.rename('DIF',inplace = True)
    DDI = DIZ -DIF
    DDI.rename('DDI',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(DIZ,left_index = True,right_index = True, how = 'left')
    output = output.merge(DIF,left_index = True,right_index = True, how = 'left')
    output = output.merge(DDI,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def TRIX_add(data_fct,N = [5,10]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:     
        # 计算中间变量
        EMA3 = ((data['Close'].ewm(span = n, adjust = False).mean()).ewm(span = n, adjust = False).mean()).ewm(span = n, adjust = False).mean()
        # 计算因子
        TRIX = EMA3/EMA3.shift(1) -1
        TRIX.rename('TRIX'+ str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(TRIX,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def Aroon_add(data_fct,N = 26):
    data = copy.deepcopy(data_fct)
    
    # 计算因子
    AroonUp = (data['Close'].rolling(N).apply(np.argmax)+1)/N
    AroonUp.rename('AroonUp',inplace = True)
    AroonDown = (data['Close'].rolling(N).apply(np.argmin)+1)/N
    AroonDown.rename('AroonDown',inplace = True)
    Aroon=AroonUp-AroonDown
    Aroon.rename('Aroon',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(Aroon,left_index = True,right_index = True, how = 'left')
    output = output.merge(AroonUp,left_index = True,right_index = True, how = 'left')
    output = output.merge(AroonDown,left_index = True,right_index = True, how = 'left')
    # 返回增加因子的数据框
    return output

def Volumn1M_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    REVS = data['Close']/data['Close'].shift(20)
    
    #计算因子
    Volumn1M = 20*data['Volume']/data['Volume'].rolling(N).sum() * REVS
    Volumn1M.rename('Volumn1M',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(Volumn1M,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def Volumn3M_add(data_fct,N1 = 5,N2 = 60):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    REVS = data['Close']/data['Close'].shift(N2)
    
    #计算因子
    Volumn3M = N2/N1*data['Volume'].rolling(N1).sum()/data['Volume'].rolling(N2).sum() * REVS
    Volumn3M.rename('Volumn3M',inplace = True)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(Volumn3M,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VEMA_add(data_fct,N = [5,10,12,26]):
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 对参数列表中的参数进行循环
    for n in N:
        # 添加VEMA
        VEMA = output['Volume'].ewm(span = n, adjust = False).mean()
        VEMA.rename('VEMA' + str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(VEMA, left_index = True, right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VMACD_add(data_fct,N = [12,26], M = 9):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 计算中间变量
    EMA1 = data['Volume'].ewm(span = N[0], adjust = False).mean()
    EMA2 = data['Volume'].ewm(span = N[1], adjust = False).mean()
    VDIFF = EMA1 - EMA2
    VDEA = VDIFF.ewm(span = M, adjust = False).mean()
    
    # 计算因子
    VMACD = 2*(VDIFF - VDEA)
    VMACD.rename('VMACD',inplace = True)
    VDIFF.rename('VDIFF',inplace = True)
    VDEA.rename('VDEA',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(VDIFF,left_index = True,right_index = True, how = 'left')
    output = output.merge(VDEA,left_index = True,right_index = True, how = 'left')
    output = output.merge(VMACD,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VOSC_add(data_fct,N = [12,26]):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    MA1 = data['Volume'].rolling(N[0]).mean()
    MA2 = data['Volume'].rolling(N[1]).mean()
    
    # 计算因子
    VOSC = (MA1 - MA2)/MA1 *100
    VOSC.rename('VOSC',inplace = True)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(VOSC,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VR_add(data_fct,N = 24):

    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    data['C_PreC'] = data['Close'] - data['Close'].shift(1)
    AV = data.apply(lambda series: series['Volume'] if series['C_PreC'] > 0 else 0,axis = 1)
    BV = data.apply(lambda series: series['Volume'] if series['C_PreC'] < 0 else 0,axis = 1)
    CV = data.apply(lambda series: series['Volume'] if series['C_PreC'] == 0 else 0,axis = 1)
    AVS = AV.rolling(N).sum()
    BVS = BV.rolling(N).sum()
    CVS = CV.rolling(N).sum()
    
    # 计算因子
    VR = (AVS + CVS/2)/(BVS + CVS/2)
    VR.rename('VR',inplace = True)
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(VR,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VROC_add(data_fct,N = [6,12]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        # 计算因子
        VROC = data['Volume']/data['Volume'].shift(n) * 100 
        VROC.rename('VROC'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(VROC,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def VSTD_add(data_fct,N = [10,20]):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    for n in N:
        # 计算因子
        VSTD = data['Volume'].rolling(n).std()
        VSTD.rename('VSTD'+str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(VSTD,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def MoneyFlow20_add(data_fct,N = 20):
    data = copy.deepcopy(data_fct)
    
    # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    # 计算中间变量
    MoneyFlow = (data['Close'] + data['High'] + data['Low'])*data['Volume']/3
    
    # 计算因子
    MoneyFlow20 = MoneyFlow.rolling(N).sum()
    MoneyFlow20.rename('MoneyFlow20',inplace = True)
    # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
    output = output.merge(MoneyFlow20,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output

def Variance_add(data_fct,N = [20,60,120]):
    data = copy.deepcopy(data_fct)
    
    # 计算中间变量
    CHGPct = data['Close']/data['Close'].shift(1) - 1

     # 制作用于最终输出的数据框copy
    output = copy.deepcopy(data_fct)
    
    for n in N:
        # 计算因子
        Variance = (CHGPct.rolling(n).std())**2 * 250
        Variance.rename('Variance' + str(n),inplace = True)
        # 将最终因子与初始数据框的深拷贝副本合并（以剔除中间变量）
        output = output.merge(Variance,left_index = True,right_index = True, how = 'left')

    # 返回增加因子的数据框
    return output


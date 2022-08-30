import pandas as pd
import numpy as np
import os,time,gc
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import sys,warnings
warnings.filterwarnings('ignore')

lbe = LabelEncoder()
mms = MinMaxScaler(feature_range=(0, 1))

ratio = 0.0   #neg为10表示负样本减少10倍,ratio==0即为submit
def find_best_threshold(y_valid, oof_prob):
    best_f2 = 0
    
    for th in tqdm([i/500 for i in range(5, 475)]):
        oof_prob_copy = oof_prob.copy()
        oof_prob_copy[oof_prob_copy >= th] = 1
        oof_prob_copy[oof_prob_copy < th] = 0

        recall = recall_score(y_valid, oof_prob_copy)
        precision = precision_score(y_valid, oof_prob_copy)
        f2 = 5*recall*precision / (4*precision+recall+1e-9)
        
        if f2 > best_f2:
            best_th = th
            best_f2 = f2
  
        gc.collect()
        
    return best_th, best_f2    
def custom_f2_eval(pred,dtrain):
    y_true = dtrain.get_label()
    num_TP = 0
    num_pred = 0
    threshold = 0.06
    for i in range(len(y_true)):
        if np.exp(pred[i,1])/(1+np.exp(pred[i,1])) >= threshold:
            num_pred += 1
            if y_true[i] == 1:
                num_TP += 1
    if num_pred == 0:return 'f2', 0.0
    if num_TP == 0:return 'f2', 0.0
    precision = num_TP / num_pred
    recall = num_TP / len(y_true[y_true == 1])
    f2 = 5 * precision * recall / (4*precision + recall)
    return 'f2',float(f2)

class xgb_ctr(object):
    def __init__(self,seed):
        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': ['logloss','auc'],
            'gamma': 1,
            'min_child_weight': 3,
            'max_depth': 8,
            'learning_rate': 0.005,
            'lambda': 0.1,
            'alpha': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.7,
            'eta': 0.05,     #0.05-0.3
            'tree_method': 'exact',
            'seed': seed,
            'nthread': 48,
            "silent": True
            }
    def train(self,df_train,df_val):
        train_x=df_train.drop(columns=['id','y'])
        train_y=df_train['y']
        print(train_x.shape,train_y.shape)
        train_matrix = xgb.DMatrix(train_x, label=train_y)    # xgb无法指定特征名称和分类特征, df里object则代表类别特征
        valid_matrix = xgb.DMatrix(df_val.drop(columns=['id','y']), label=df_val['y'])
         #-----------
        watchlist = [(valid_matrix, 'eval')]
        if ratio>0:
            self.model = xgb.train(self.params, train_matrix, num_boost_round=3000, evals=watchlist , early_stopping_rounds=1000, verbose_eval=20)  #,feval=f2_score,maximize=True
            print("\n".join(("%s: %.2f" % x) for x in list(sorted( self.model.get_fscore().items(),key=lambda x:x[1],reverse=False)) ))
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(figsize=(15,15))
            plot_importance(self.model, height=0.5, ax=ax, max_num_features=64)
            plt.show()
            print("最优迭代步:",self.model.best_iteration,self.model.best_ntree_limit,self.model.best_score)
            return self.evaluate(df_val)
        else:
            self.model = xgb.train(self.params, train_matrix, num_boost_round=3000, evals=watchlist, early_stopping_rounds=20000, verbose_eval=50)  #不加伪标签时2300跑一个submit_xgb_0.csv重命名为submit_all.csv，然后加进去用3000跑带伪标签的结果。
            print("\n".join(("%s: %.2f" % x) for x in list(sorted( self.model.get_fscore().items(),key=lambda x:x[1],reverse=False)) ))
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(figsize=(15,15))
            plot_importance(self.model, height=0.5, ax=ax, max_num_features=64)
            plt.show()
            print("最优迭代步:",self.model.best_iteration,self.model.best_ntree_limit,self.model.best_score)
            return 0.26,0,0,0
    def evaluate(self,df):
        val_x=df.drop(columns=['id','y'])
        labels=df['y']
        valid_matrix = xgb.DMatrix(val_x, labels)
        logits = self.model.predict(valid_matrix, ntree_limit = self.model.best_ntree_limit)    #自带度量里没有f2，故此处不用, ntree_limit=model.best_ntree_limit
        auc = roc_auc_score(labels.values,logits)
        print("valid_auc:",auc)
        best_th, best_f2 = find_best_threshold(labels, logits)
        print(best_th, best_f2)
        y = logits>best_th
        print(y.sum())
        precision = precision_score(labels,y)
        recall = recall_score(labels,y)
        f2_score = 5*recall*precision/(4*precision+recall+1e-9)   #更注重recall，尽可能不放过任何一个正例
        return best_th,f2_score,precision,recall
    def predict(self,df):
        #测试集
        test_x=df.drop(columns=['id'])
        test_matrix = xgb.DMatrix(test_x)
        logits= self.model.predict(test_matrix, ntree_limit = self.model.best_ntree_limit)
        return logits
        
def fill(df):
    #df['j6'] = df['j6'].apply(lambda x:x+1)   #'k6','l6','m6'
    df = df.fillna(-999)
    return df

if __name__ == "__main__":
    start_time=time.time()

    xy_train = pd.read_csv("./Data_A/xy_train.csv",low_memory=False)
    xy_valid = pd.read_csv("./Data_A/xy_test_A.csv",low_memory=False)
    x_test = pd.read_csv("./Data_A/x_test.csv")
    y_test = pd.read_csv("submit_all.csv")
    x_test = x_test.merge(y_test, how='left', on='id')
    #x_test['y'] = -1
    print(xy_train.tail())

    #控制正负样本比例后，与test粘在一起做lbe和mms，粘的时候不能merge上y
    #按照缺失度由小到大排序，进行负采样
    #neg_df = xy_train[xy_train['y']==0].reset_index(drop=True)
    #neg_index = neg_df.isnull().sum(axis=1).to_frame().sort_values(0).index.tolist()
    #neg_df = neg_df.loc[neg_index[:neg_df.shape[0]//neg]]
    #xy_train = pd.concat((xy_train[xy_train['y']==1],neg_df)).sample(frac=1,random_state=42).reset_index(drop=True)
    
    data = pd.concat((xy_train,xy_valid,x_test)).reset_index(drop=True)
    #data = fill(data)
    train_Fea = data.drop(columns=['core_cust_id','prod_code','date']).columns.tolist()
    #+[f'f{i+2}' for i in range(5)]+[f'f_{i+2}' for i in range(5)]
    
    data = data[train_Fea]   #特征过滤，只留下树模型能处理的 Dense和Sparse
    drop_cols = [c for c in train_Fea if data[c].dtype != 'object' and data[c].std() == 0]
    data.drop(drop_cols, axis=1, inplace=True)
    print("用方差drop后",drop_cols)

    xy_train,xy_valid = data.iloc[:xy_train.shape[0]],data.iloc[xy_train.shape[0]:-x_test.shape[0]].reset_index(drop=True)
    x_test = data.iloc[-x_test.shape[0]:].reset_index(drop=True)

    print("使用伪标签前",xy_train.shape, xy_valid.shape)
    if ratio==0:xy_train = pd.concat((xy_train,xy_valid)).reset_index(drop=True)   #决定是否线上添加伪标签,x_test[x_test['y']==0]
    else:
        model = xgb_ctr(42)
        best_th,score,presision,recall = model.train(xy_train,xy_valid)
        print( "val-f2-score：",score,"val-presision：",presision,"val-recall：",recall )   
        print("耗时(min):",(time.time()-start_time)/60)
        x_valid = xy_valid.drop(columns=['y'])
        valid_pred = model.predict(x_valid)
        preds = np.zeros([xy_valid.shape[0]])
        while(preds.sum()<270000):    #保证recall比较高
            preds = valid_pred.copy()
            preds[preds>=best_th],preds[preds<best_th] = 1,0
            print(best_th,"valid预测出的正例数量：",preds.sum())
            best_th = best_th-0.0002
        x_valid['y'] = preds
        xy_train = pd.concat((xy_train,x_valid[x_valid['y']==0])).reset_index(drop=True)
    print("使用伪标签后",xy_train.shape, xy_valid.shape)
    
    xy_train = xy_train.sample(frac=1,random_state=42).reset_index(drop=True)
    x_test = x_test.drop(columns=['y'])
    
    preds = pd.DataFrame()
    for seed in [0,24,42,128,1024,1996,1998,1999,2020,2022]:   #0,24,42,128,1024,1996,1998,1999,2020,2022
        print(seed)
        model = xgb_ctr(seed)
        best_th,score,presision,recall = model.train(xy_train,xy_valid)
        print( "val-f2-score：",score,"val-presision：",presision,"val-recall：",recall )            
        pred = model.predict(x_test)
        pred = pd.DataFrame(pred,columns=[f'xgb{seed}'])
        preds = pd.concat((preds,pred),axis=1)
    preds.to_csv("preds_xgb.csv",index=False)
    preds0 = preds.mean(axis=1).values
    #preds0 = preds.max(axis=1).values
    preds = np.zeros([x_test.shape[0]])
    while(preds.sum()<20000):    #保证recall比较高
        preds = preds0.copy()
        preds[preds>=best_th],preds[preds<best_th] = 1,0
        print(best_th,"预测出的正例数量：",preds.sum())
        best_th = best_th-0.001
    df = x_test['id'].to_frame()
    df['y'] = preds
    df.to_csv(f"submit_xgb_{score}.csv",index=False)
    print("耗时(min):",(time.time()-start_time)/60)

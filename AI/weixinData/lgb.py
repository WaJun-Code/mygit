import pandas as pd
import numpy as np
import os,time
import lightgbm
import xgboost as xgb
from deepfm import uAUC
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler
from scipy import sparse
import sys
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

class lgb_ctr(object):
    '''
    '''
    def __init__(self,action):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'boost_from_average': True,
            'train_metric': True,
            'feature_fraction_seed': 1,
            'learning_rate': 0.005,
            'is_unbalance': True,  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
            'num_leaves': 1024,  # 一般设为少于2^(max_depth)
            'max_depth': -1,  # 最大的树深，设为-1时表示不限制树的深度
            'min_child_samples': 32,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
            'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
            'subsample': 0.8,  # 训练实例的子样本Subsample比率
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 0.1,  # L1 regularization term on weights
            'reg_lambda': 0.5,  # L2 regularization term on weights
            'nthread': 12,
            'verbose': 0,
            'force_row_wise':True
        }
        self.action=action
        self.select_frts=FEA_FEED_LIST
        #feed embedding by PCA
        #self.select_frts+=['feed_embed_'+str(i) for i in range(32)]
    def train(self,df_train):
        splitnum=int((1-0.25)*train.shape[0])
        df_val=df_train.iloc[splitnum:].reset_index(drop=True)
        df_train=df_train.iloc[:splitnum].reset_index(drop=True)
        df_train=df_train.reset_index(drop=True)    #此处是否留验证集

        train_x=df_train[self.select_frts]
        train_y=df_train[self.action]
        print(train_x.shape,train_y.shape)
        train_matrix = lightgbm.Dataset(train_x, label=train_y)
         #-----------
        self.model=lightgbm.train(self.params, train_matrix
                    ,num_boost_round=200)
        print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(list(train_x.columns), self.model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:5]))
        return self.evaluate(df_val)
    def evaluate(self,df):
        #测试集
        test_x=df[self.select_frts].values
        labels=df[self.action].values
        userid_list = df['userid'].astype(str).tolist()
        logits = self.model.predict(test_x)
        uauc=uAUC(labels,logits,userid_list)
        return df[["userid","feedid"]],logits,uauc
    def predict(self,df):
        #测试集
        test_x=df[self.select_frts].values
        logits= self.model.predict(test_x)
        return df[["userid","feedid"]],logits

if __name__ == "__main__":
    t=time.time()
    submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
    score=0

    feed_info_df = pd.read_csv(FEED_INFO)
    name=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list','machine_tag_list','description']
    wordcol=[f"word{j}"for j in range(8*len(name))]
    #此处改embedding维度
    columns,usercol,authorcol=[f"embed{i}" for i in range(32)],[f"uembed{i}" for i in range(8)],[f"aembed{i}" for i in range(8)]
    labelCol=[f"ulabel{j}" for j in range(16)]+[f"flabel{j}" for j in range(16)]+[f"alabel{j}" for j in range(16)]
    
    for action in ACTION_LIST:
        USE_FEAT = ['userid', action] + FEA_FEED_LIST
        target = [action]
        data = pd.read_csv(ROOT_PATH + f'/train_test_data_for_{action}.csv')

        # columns=[]
        # usercol,authorcol=[],[]
        USE_FEAT= USE_FEAT+labelCol +columns+usercol+authorcol+wordcol
        
        train,test= data.iloc[:-submit.shape[0]].reset_index(drop=True), data.iloc[-submit.shape[0]:].reset_index(drop=True)
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])
        train,test= data.iloc[:-submit.shape[0]].reset_index(drop=True), data.iloc[-submit.shape[0]:].reset_index(drop=True)
        train = train.sort_values('date_')   #按照日期排序，防止数据泄露
        train = train[USE_FEAT]
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])

        test = test[[i for i in USE_FEAT if i != action]]
        test[target[0]] = 0
        test = test[USE_FEAT]

        model = lgb_ctr(action)
        ids,logits,action_auc=model.train(train)
        print(action,action_auc)
        if action=='read_comment':
            score+=0.4*action_auc
        elif action=='like':
            score+=0.3*action_auc
        elif action=='click_avatar':
            score+=0.2*action_auc
        else:
            score+=0.1*action_auc
        ids,logits=model.predict(test)
        pred_ans = logits
        submit[action] = pred_ans
    print('Time cost: %.2f s'%(time.time()-t))
    print("验证集上得分score=",score)
    # 保存提交文件
    submit.to_csv("./submit_base_lgb.csv", index=False)
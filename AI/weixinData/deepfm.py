# -*- coding: utf-8 -*-
import time
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *

from sklearn.metrics import roc_auc_score
from collections import defaultdict
import re
def strcut(x):
    y=[]
    y0=re.split(' |;',str(x).strip())
    try:
        for i in range(len(y0)):
            if (i+1)%2==0:
                if float(y0[i])>0.01:y.append(float(y0[i-1]))
    except:
        y=[0]
    return y
def get_col(feedinfo,name):
    collist,cols=[],[]
    for xx in name:
        col=[]
        if xx=='machine_tag_list':n=feedinfo[xx].apply(strcut).apply(len).max()
        else:n=feedinfo[xx].apply(lambda x:[float(i) for i in re.split(' |;',str(x).strip())]).apply(len).max()
        for i in range(n):col.append(xx+str(i))
        cols.append(col)
        collist+=col
    return collist,cols
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc

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

class MyBaseModel(BaseModel):

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        user_id_list=x['userid']
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            user_id_list=user_id_list.iloc[split_at:].reset_index(drop=True)
        else:
            val_x = []
            val_y = []
            user_id_list=None
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y,user_id_list, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                else:epoch_logs["val_auc"]=0
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        
        callbacks.on_train_end()

        return self.history,epoch_logs["val_auc"]

    def evaluate(self, x, y,user_id_list, batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)

        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                if name == 'auc':
                    eval_result[name] = uAUC(y,pred_ans,user_id_list)
                else:
                    eval_result[name] = metric_fun(y, pred_ans)
            except:
                eval_result[name] =0

        return eval_result

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

class MyDeepFM(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(1024,128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

if __name__ == "__main__":
    t=time.time()
    submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
    score=0

    name=['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list','machine_tag_list','description']
    wordcol=[f"word{j}"for j in range(8*len(name))]
    #此处改embedding维度
    columns,usercol,authorcol=[f"embed{i}" for i in range(64)],[f"uembed{i}" for i in range(8)],[f"aembed{i}" for i in range(8)]
    labelCol=[f"ulabel{j}" for j in range(16)]+[f"flabel{j}" for j in range(16)]+[f"alabel{j}" for j in range(16)]

    for action in ACTION_LIST:
        target = [action]
        data = pd.read_csv(ROOT_PATH + f'/train_test_data_for_{action}.csv')

        # columns=[]
        # usercol,authorcol=[],[]
        # labelCol=[]

        nsplit=0.0
        if action=='forward':
            epochs=2
        elif action=='click_avatar':
            epochs,columns,usercol,authorcol,labelCol=2,[],[],[],[]
        else:epochs=1

        USE_FEAT= ['userid', action] + FEA_FEED_LIST +labelCol +columns+usercol+authorcol+wordcol
        dense_features = ['videoplayseconds']+labelCol+columns+usercol+authorcol+wordcol

        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]
        # 2.count nunique features for each sparse field,and record dense feature field name
        sparseNum={}
        for feat in sparse_features:
            sparseNum[feat]=data[feat].nunique()
            if feat=='userid':sparseNum[feat]=20000
            elif feat=='feedid':sparseNum[feat]=112872
            elif feat=='authorid':sparseNum[feat]=18789

        fixlen_feature_columns = [SparseFeat(feat, sparseNum[feat], embedding_dim=4)
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)


        train,test= data.iloc[:-submit.shape[0]].reset_index(drop=True), data.iloc[-submit.shape[0]:].reset_index(drop=True)
        # train=pd.concat((train[train[action]==0].sample(frac=0.7,random_state=0,replace=False),train[train[action]==1]))  #控制负样本比例
        train = train.sort_values('date_')   #按照日期排序，防止数据泄露
        train = train[USE_FEAT]
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])

        test = test[[i for i in USE_FEAT if i != action]]
        test[target[0]] = 0
        test = test[USE_FEAT]

        # 3.generate input data for model
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       task='binary',
                       l2_reg_embedding=1e-1, device=device)

        model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        history,action_auc = model.fit(train_model_input, train[target].values, batch_size=4096, epochs=epochs, verbose=1,
                            validation_split=nsplit)
        
        print(action,action_auc)
        #计算验证集上的score
        if action=='read_comment':
            score+=0.4*action_auc
        elif action=='like':
            score+=0.3*action_auc
        elif action=='click_avatar':
            score+=0.2*action_auc
        else:
            score+=0.1*action_auc

        pred_ans = model.predict(test_model_input, 1024)
        submit[action] = pred_ans
        torch.cuda.empty_cache()
    print('Time cost: %.2f s'%(time.time()-t))
    print("验证集上得分score=",score)
    # 保存提交文件
    submit.to_csv("./submit_base_deepfm.csv", index=False)
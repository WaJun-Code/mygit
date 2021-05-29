lr,epochs,batchsize=0.00001,1,4
import re
# def clean_html(text):
#     pattern = re.compile(r'<[^>]+>',re.S)
#     result = pattern.sub('', text)
#     result="".join(result.split())
#     return result
import pandas as pd
import numpy as np
company_name = pd.read_excel('2_公司实体汇总_20210414_A1.xlsx', names=['name'])
train_data = pd.read_excel('3_训练集汇总_20210414_A1.xlsx')
train_data.drop(0, inplace=True, axis=0)
train_data=train_data.fillna('/')
indexLabel=train_data['LABEL'].value_counts().to_frame().index
def replace_postfix(s):
    for x in '有限责任公司 股份有限公司 控股股份公司 总公司 有限公司 投资合伙企业 子公司 公司 集团'.split(' '):
        if s[-len(x):] == x:    
            return s[:-len(x)], x
        
    return s, None   #不在以上词条内的即为None
company_name['name2'] = company_name['name'].apply(replace_postfix)
company_name['name_short'] = company_name['name2'].apply(lambda x: x[0])
company_name['name_postfix'] = company_name['name2'].apply(lambda x: x[1])
train_data = train_data[['NEWS_BASICINFO_SID','NEWS_TITLE', 'CONTENT','COMPANY_NM', 'LABEL']]

from sklearn.preprocessing import LabelEncoder
train_data = train_data.sample(frac = 1.0)   #frac要抽取行的比例
lbl = LabelEncoder().fit(train_data['LABEL'])
train_data['LABEL'] = lbl.transform(train_data['LABEL'])
indexLabel=pd.core.frame.DataFrame(indexLabel.tolist(),index=train_data['LABEL'].value_counts().to_frame().index.tolist()).T
import re
def process(s):
    s=re.findall(r'([\u2E80-\u9FFF]+)',s)   #或者用r'([\u2E80-\u9FFF]+)'保留中文标点符号，用r'([\u4E00-\u9FA5]+)'则只留中文
    return ' '.join(list(s))
train_data['CONTENT']=train_data['CONTENT'].apply(process)

import jieba
def strcut(s):
    seg_list = jieba.cut(s)
    return ' '.join(list(seg_list))
train_data['NEWS_TITLE']=train_data['NEWS_TITLE'].apply(strcut)
train_data['CONTENT']=train_data['CONTENT'].apply(strcut)
data=train_data['NEWS_TITLE']+train_data['CONTENT']
from sklearn.model_selection import train_test_split
tr_x, val_x, tr_y, val_y = train_test_split(
    data, train_data['LABEL'],
    stratify = train_data['LABEL'],
    test_size=0.05
)
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(ngram_range=(1,1))
# train_title_ttidf = tfidf.fit_transform(data)
# print(train_title_ttidf.shape)

# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import RidgeClassifier
# clf = RidgeClassifier()
# tr_x, val_x, tr_tfidf, val_tfidf, tr_y, val_y = train_test_split(
#     data, train_title_ttidf, train_data['LABEL'],
#     stratify = train_data['LABEL'],
#     test_size=0.2
# )
# clf.fit(tr_tfidf, tr_y)
# print(clf.score(val_tfidf, val_y))

test_data = pd.read_excel('测试集-1.xlsx',nrows=22288)
test_data=test_data.fillna('/')
test_data['LABEL'] = 0
test_data = test_data[['NEWS_BASICINFO_SID','NEWS_TITLE', 'CONTENT','LABEL']]
test_data['CONTENT']=test_data['CONTENT'].apply(process)
test_data['NEWS_TITLE']=test_data['NEWS_TITLE'].apply(strcut)
test_data['CONTENT']=test_data['CONTENT'].apply(strcut)
input_test=test_data['NEWS_TITLE']+test_data['CONTENT']

#Bert分类
import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
input_test=tokenizer(list(input_test), truncation=True, padding=True, max_length=128)
train_encoding = tokenizer(list(tr_x), truncation=True, padding=True, max_length=128)
val_encoding = tokenizer(list(val_x), truncation=True, padding=True, max_length=128)
from torch.utils.data import Dataset, DataLoader, TensorDataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)
# class textTest(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
        
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item

#     def __len__(self):
#         return 1
train_dataset = TextDataset(train_encoding, tr_y)
test_dataset = TextDataset(val_encoding, val_y)
input_dataset=TextDataset(input_test,test_data['LABEL'])
input_dataset= DataLoader(input_dataset, batch_size=1, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

from transformers import BertForSequenceClassification 
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=13)
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optim = AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * 1
loss_function = torch.nn.CrossEntropyLoss()
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = torch.tensor(batch['labels'],dtype=torch.long).to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # loss = outputs[0]
        
        loss = loss_function(outputs[1], labels)
        
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))
    
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = torch.tensor(batch['labels'],dtype=torch.long).to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(test_dataloader)))
    print("-------------------------------")
def predict_test():
    model.eval()
    In=0
    for inputx in input_dataset:
        with torch.no_grad():
            input_ids = inputx['input_ids'].to(device)
            attention_mask = inputx['attention_mask'].to(device)
            labels = torch.tensor(inputx['labels'],dtype=torch.long).to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()        
        with open('labelResult.txt','a') as fp:
            fp.writelines([str(test_data['NEWS_BASICINFO_SID'].iloc[In]),'\t',indexLabel[np.argmax(logits[0], axis=0)][0],'\n'])
        In=In+1
    print(In)
torch.cuda.empty_cache()
import time
start_time=time.time()
for epoch in range(epochs):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
print('训练耗时（min）:',(time.time()-start_time)/60)
start_time=time.time()
predict_test()
print('录入耗时（min）:',(time.time()-start_time)/60)
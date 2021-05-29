tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']

import torch

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = torch.load('bert-ner-4.pt', map_location=device)

def predict(s):
    item = tokenizer([s], truncation=True, padding='longest', max_length=64)
    with torch.no_grad():
        input_ids = torch.tensor(item['input_ids']).to(device).reshape(1, -1)
        attention_mask = torch.tensor(item['attention_mask']).to(device).reshape(1, -1)
        labels = torch.tensor([0] * attention_mask.shape[1]).to(device).reshape(1, -1)

        outputs = model(input_ids, attention_mask, labels)
        outputs = outputs[0].data.cpu().numpy()

    outputs = outputs[0].argmax(1)[1:-1]
    ner_result = ''
    ner_flag = ''
    res = {'机构':[], '人名':[], '位置':[]}

    for o, c in zip(outputs, s):
        # 0 就是 O，没有含义
        if o == 0 and ner_result == '':
            continue

        #
        elif o == 0 and ner_result != '':
            if ner_flag == 'O':
                res['机构'].append(ner_result)
            if ner_flag == 'P':
                res['人名'].append(ner_result)
            if ner_flag == 'L':
                res['位置'].append(ner_result)
            ner_result = ''
        elif o != 0:
            ner_flag = tag_type[o][2]
            ner_result += c
    return res


# s = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
# predict(s)

import pandas as pd
import numpy as np
import jieba
company_name = pd.read_excel('RISKLABEL_Training/2_公司实体汇总_20210414_A1.xlsx', names=['name'])
test_data = pd.read_excel('测试集-1.xlsx', nrows=22288)
# test_data = pd.read_excel('RISKLABEL_Training/3_训练集汇总_20210414_A1.xlsx')
# test_data.drop(0, inplace=True, axis=0)
# test_data = test_data[['NEWS_TITLE', 'COMPANY_NM', 'LABEL']]
test_data=test_data.fillna('/')
def replace_postfix(s):
    for x in '有限责任公司 股份有限公司 控股股份公司 总公司 有限公司 投资合伙企业 子公司 公司 集团'.split(' '):
        if s[-len(x):] == x:    
            return s[:-len(x)], x
        
    return s, None   #不在以上词条内的即为None
company_name['name2'] = company_name['name'].apply(replace_postfix)
company_name['name_short'] = company_name['name2'].apply(lambda x: x[0])
company_name['name_postfix'] = company_name['name2'].apply(lambda x: x[1])


import re
def simplify(s):
    s=re.findall(r'([\u2E80-\u9FFF]+)',s)   #或者用r'([\u2E80-\u9FFF]+)'保留中文标点符号，用r'([\u4E00-\u9FA5]+)'则只留中文
    return ''.join(list(s))
test_data['CONTENT']=test_data['CONTENT'].apply(simplify)
data=test_data['NEWS_TITLE']+test_data['CONTENT']



def process(s):
    match1 = company_name[company_name['name_short'].apply(lambda x: x in row)]
    if match1.shape[0] > 0:
        match1.loc[:, 'name_len'] = match1['name_short'].apply(len)
        match1 = match1.sort_values(by='name_len')
        return match1.iloc[-1]['name']

    out = predict(s)['机构']
    out = [x  for o in out for x in jieba.cut(replace_postfix(o)[0], cut_all=True)] + [x  for o in out for x in jieba.cut(replace_postfix(o)[0], cut_all=False)]
    names  = []
    for o in out:
        index = company_name['name_short'].apply(lambda x: o in x)
        num = index.sum()
        if 0 < num < 50:  # 根据词频数进行一轮筛选
            name = np.array(company_name[index]['name']).tolist()  # 得到公司名字
            names += name
    if len(names) > 0:
        result = pd.core.frame.DataFrame(names)[0].value_counts()  # 按照出现次数降序排列
        result = result.to_frame().index[0]
        return result
    return '/'


import time
start_time=time.time()
In=0

f=open('TestNERResult2.csv','w')
f.close()
for row in data:
    In = In + 1
    c_n = process(row)
    with open('TestNERResult2.csv', 'a') as fp:
        fp.writelines([str(test_data['NEWS_BASICINFO_SID'].iloc[In-1]), ',', c_n, '\n'])
    print('进度:{}'.format(In))
print('耗时（min）:', (time.time() - start_time) / 60)
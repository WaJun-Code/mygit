tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('bert-ner-8.pt', map_location=device)
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
# print(predict(s))

import pandas as pd
import numpy as np
lose0='他是 一定 控制 有限责任公司 股份有限公司 控股股份公司 总公司 有限公司 投资合伙企业 投资有限公司 子公司 公司 集团 控制人 股东 实际'.split(' ')
company_name = pd.read_excel('2_公司实体汇总_20210414_A1.xlsx', names=['name'])
test_data = pd.read_excel('测试集-1.xlsx',nrows=22288)
test_data=test_data.fillna('/')
def replace_postfix(s):
    for x in lose0:
        if s[-len(x):] == x:    
            return s[:-len(x)], x
        
    return s, None   #不在以上词条内的即为None
company_name['name2'] = company_name['name'].apply(replace_postfix)
company_name['name_short'] = company_name['name2'].apply(lambda x: x[0])
company_name['name_postfix'] = company_name['name2'].apply(lambda x: x[1])
test_data = test_data[['NEWS_BASICINFO_SID','NEWS_TITLE', 'CONTENT']]
import re
def process(s):
    s=re.findall(r'([\u2E80-\u9FFF]+)',s)   #或者用r'([\u2E80-\u9FFF]+)'保留中文标点符号，用r'([\u4E00-\u9FA5]+)'则只留中文
    s=' '.join(list(s))
    return s
test_data['CONTENT']=test_data['CONTENT'].apply(process)
test_data['NEWS_TITLE']=test_data['NEWS_TITLE'].apply(process)

epsIndex,eps,loseNum,ComNum=2,2,100,79   #loseNum是前loseNum个高频词丢弃150时最低为45,ComNum是当匹配结果大于ComNum时丢弃

import jieba
def strcut(s):
    seg_list = jieba.cut(s)
    return list(seg_list)
word=[]
for wordi in company_name['name_short'].apply(strcut):
    word=word+wordi
word=pd.core.frame.DataFrame(word)
word_num=word[0].value_counts().to_frame().T
lose='中国 美国 兰州 南京 苏州 九龙 香港 澳门 台湾 北京 上海 天津 重庆 内蒙古 新疆 广西 西藏 宁夏 辽宁 吉林 黑龙江 河北 山西 陕西 甘肃 青海 山东 安徽 江苏 浙江 河南 湖北 湖南 江西 台湾 福建 云南 海南 四川 贵州 广东 成都 武汉 贵阳 杭州 广州 西安 乌鲁木齐 拉萨 昆明 哈尔滨 青岛'.split(' ')  #前100个高频词汇组成的dataframe, 最低79个重复


def getstr(df):
    a=[]
    for idx,_ in df.iterrows():
        if idx>0:   #第一行的num不读取
            a=a+np.array(df.iloc[idx,:]).tolist()
    i=0
    while i<len(a):
        if a[i]==None:
            del(a[i])
            i=i-1
        i=i+1
    return a

def str_pop(s):
    i,k=0,[]
    for aa in s:
        if len(aa)<2 or aa in lose0:
            k.append(i)
        i=i+1
    for i in range(len(k)):
        s.pop(k[len(k)-i-1])
    return s

def get_result(s):   #输入分词好后的一个list
    NmNum,strs=[],[]
    for i in range(1,len(s)-1):
        if s[i] in lose or len(s[i])==1:      #当分词在高频词汇里，则将之前后组合
            s.append(s[i-1]+s[i])
            s.append(s[i]+s[i+1])
    s=str_pop(s)      #删掉只有一个字的词
    for i in range(len(s)):
        index=company_name['name_short'].apply(lambda x: s[i] in x)  #得到匹配成功公司数量
        num=index.sum()
        if 0<num<ComNum:   #根据词频数进行一轮筛选
            name=np.array(company_name[index]['name']).tolist()  #得到公司名字
            strs.append(s[i])
            NmNum.append([num]+name)
    if NmNum==[]:return pd.core.frame.DataFrame([0],index=[''])
    NmNum=pd.core.frame.DataFrame(NmNum, index=strs).sort_values(0).T
    result=getstr(NmNum)
    result=pd.core.frame.DataFrame(result)[0].value_counts().to_frame()  #按照出现次数降序排列
    return result

def content_match(row):
    result=get_result(strcut(row))
    if len(result.index.tolist())==1:return result.index[0]
    if result.iloc[0][0]!=result.iloc[1][0]:   #可以0和1、2看想要的误差
        match1 = result.index[0]                      #不可能为空

    elif result.iloc[1][0]>eps:  #有频数相同情况
        match1 = result.index[1]                #不可能为空
    else:
        match1 = ''    #大于2个结果小于eps频数则无法选择，置为空
    return match1

def get_match(row):   #先通过nameshort直接匹配，若非唯一结果则转入下一步
    match1 = company_name[company_name['name_short'].apply(lambda x: x in row)]
    if 0<match1.shape[0] <5:
        match1['L']=match1['name_short'].apply(len)
        match1=match1.sort_values(by='L')
        match1=match1.iloc[-1]['name']
    else:
        result=get_result(strcut(row))
        if result.index[0]!='' and result.shape[0]<5:  #拆分title后获得小于5匹配的，取最短的公司名
            match1 = result.index.to_frame(index=0)                       #可能为空
            match1['L']=match1[0].apply(len)
            match1=match1.sort_values(by='L')
            match1=match1.iloc[0,0]
        else:        #否则引入content辅助匹配
            row=row+' '+test_data['CONTENT'].iloc[In]
            #用NER压缩信息
            nerow='a'
            if len(test_data['CONTENT'].iloc[In])>400:
                x=''
                nerow=predict(row)['机构']+predict(row)['位置']
                for i in nerow:x=x+' '+i
                nerow=x
            match1=content_match(nerow)
            
            if match1=='':
                match1=content_match(row)
    return match1
      

from tqdm import tqdm_notebook
import time
start_time=time.time()
In,acc=0,0
list1=0
for row in tqdm_notebook(test_data['NEWS_TITLE'][In:]):
    match1=get_match(row)
    if match1=='':
        list1=list1+1
        match1='/'
    else:
        acc=acc+1
    with open('companyNmTestResult.txt','a') as fp:
        fp.writelines([str(test_data['NEWS_BASICINFO_SID'].iloc[In]),'\t', match1 ,'\n'])

    In=In+1
    if In==100:break
    print('进度:',In/22288)
print('已正确识别的公司数：',acc)
print('正确识别的公司数（总22288）：',list1,acc,In-list1)
print('耗时（min）:',(time.time()-start_time)/60)
import json,re

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

with open("labels0.txt", 'r', encoding='utf8') as f:s0 = f.readlines()
s0 = "".join(s0)
s0 = "".join(re.findall('[\u4e00-\u9fa5]',s0) )
print(len(s0))

class DADataSet(Dataset):
    def __init__(self, args):
        with open(args.data_path, 'r', encoding='utf8') as f:
            self.lines = f.readlines()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        with open("./labels.txt", 'r', encoding='utf8') as f:
            labels = f.readlines()
        label_icd = []
        label2id = {}
        for idx, label in enumerate(labels):
            label_icd.append(label.strip())
            label2id[label.strip()] = idx
        self.label2id = label2id

    def __getitem__(self, idx):
        #可以使用tf-idf，对字或者jieba分词后的数据进行建模，但要考虑1w个句子里没有包含的词句如何填充
        #history_of_present_illness出现血压就都判一个["高血压","否认高血压"]
        def supplementary_cut(x): #先删掉“否认”"无"部分，再直接匹配
            L = set()
            x = x.strip().replace("，","。").split("。")
            x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
            n = len(x)
            for i in range(n):
                if "示" in x[i]:
                    L.add(x[i])
            return "。".join(L)
        def past_cut(x): #先删掉“否认”"无"部分，再直接匹配
            L = set()
            x = x.strip().replace("，","。").split("。")
            #删掉含“血压”的句子
            i = 0
            while i<len(x):
                if "否认" in x[i] or "无" in x[i]:
                    x.pop(i)
                    continue
                i+=1
            x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
            n = len(x)
            for i in range(n):
                if "有" in x[i] or "诊断" in x[i]:
                    L.add(x[i])
            return "。".join(L)
        def present_cut(x):   #df["history_of_present_illness"].apply(str_cut)
            L = set()
            x = x.strip().replace("，","。").split("。")
            #删掉含“血压”的句子
            i = 0
            while i<len(x):
                if "血压" in x[i]:
                    x.pop(i)
                    continue
                i+=1
            x = ["".join(re.findall('[\u4e00-\u9fa5]',i)) for i in x]
            n = len(x)
            for i in range(n):
                if "收" in x[i]:
                    if i>0 and "收" not in x[i-1]:L.add(x[i-1])
                    L.add(x[i])
                    if i<n-1 and "收" not in x[i+1]:L.add(x[i+1])
                elif "诊" in x[i]:
                    L.add(x[i])
                    if i<n-1 and "诊" not in x[i+1]:L.add(x[i+1])
            return "。".join(L)

        item = self.lines[idx]
        data = json.loads(item)

        data["history_of_present_illness"] = present_cut(data["history_of_present_illness"])
        data["past_history"] = past_cut(data["past_history"])
        if data["supplementary_examination"] == None:data["supplementary_examination"] = "暂缺"   #零填
        data["supplementary_examination"] = supplementary_cut(data["supplementary_examination"])
        data["chief_complaint"] = "。".join(re.findall('[\u4e00-\u9fa5]+',data["chief_complaint"]) )
        data["physical_examination"] = "。".join(re.findall('[\u4e00-\u9fa5]+',data["physical_examination"]) )
        
        present_illness = data["history_of_present_illness"]+data["past_history"] +data["supplementary_examination"]+data["chief_complaint"]+data["physical_examination"]
        
        input_ids, attn_mask = self._get_token( s0+present_illness, 512)   #353
        targets = data['diagnosis']
        label = self._get_label_ids(targets)
        return input_ids, attn_mask, label

    def __len__(self):
        return len(self.lines)

    def _get_token(self, content, pad_size):
        all_tokens = self.tokenizer.encode_plus(content, max_length=pad_size, padding="max_length", truncation=True)
        input_ids = torch.LongTensor(all_tokens['input_ids']).to(self.args.device)
        attention_mask = torch.LongTensor(all_tokens['attention_mask']).to(self.args.device)
        return input_ids, attention_mask

    def _get_label_ids(self, targets):
        label = [0] * len(self.label2id)
        for target in targets:
            if target in self.label2id:
                idx = self.label2id[target]
                label[idx] = 1
        label = torch.FloatTensor(label).to(self.args.device)
        return label

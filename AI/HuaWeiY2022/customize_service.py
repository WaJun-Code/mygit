# -*- coding: utf-8 -*-
import logging
import os

import torch
import json,re,jieba
from transformers import BertTokenizer

from model_service.pytorch_model_service import PTServingBaseService
from model import BaselineModel

logger = logging.getLogger(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

s0 = ['贫血', '肺炎', '结肠', '结核', '子宫内膜', '带状疱疹', '支气管', '糖尿病', '疱疹性咽峡', '心肌', '便秘', '腰肌劳损', '高血压', '前列腺增生', '肺气肿', '高脂血症', '口腔', '咽炎', '鼻窦', '上呼吸道', '脑卒中', '腹泻', '肾小球', '眩晕综合征', '结膜', '急性鼻咽', '腹痛', '短暂性脑缺血发作', '输卵管', '骨折', '先天性心脏病', '关节', '腹股沟疝', '椎动脉型颈椎', '痔', '胆囊', '慢性肺源性', '皮炎', '肛周脓肿', '急性扁桃体', '泌尿系结石', '中耳', '咽喉', '阑尾', '胆管结石', '腰椎间盘突出', '偏头痛', '泌尿道', '胃肠', '冠状动脉粥样硬化性', '肩周', '鼻出血']

class DaService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingBaseService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(model_path))
        bert_path = os.path.join(dir_path, 'chinese-bert-wwm-ext')
        self.model = BaselineModel(bert_path=bert_path)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        label_path = os.path.join(dir_path, 'labels.txt')
        self.id2label = []
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f:
                self.id2label.append(line.strip())

    def _get_token(self, content, pad_size):
        all_tokens = self.tokenizer.encode_plus(content, max_length=pad_size, padding="max_length", truncation=True)
        input_ids = torch.LongTensor(all_tokens['input_ids']).to(DEVICE)
        attention_mask = torch.LongTensor(all_tokens['attention_mask']).to(DEVICE)
        return input_ids, attention_mask

    def _get_diagnosis(self, pred):
        pred_index = [i for i in range(len(pred)) if pred[i] == 1]
        pred_diagnosis = [self.id2label[index] for index in pred_index]
        return pred_diagnosis

    def _preprocess(self, data):
        # https两种请求形式
        # 1. form-data文件格式的请求对应：data = {"请求key值":{"文件名":<文件io>}}
        # 2. json格式对应：data = json.loads("接口传入的json体")
        data_dict = data.get('json_line')
        for v in data_dict.values():
            infer_dict = json.loads(v.read())
            return infer_dict

    def _inference(self, data):
        self.model.eval()
        emr_id = data.get('emr_id')
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
        def post_process(x):
            post_out = torch.zeros([len(s0),])
            for i in range(len(s0)):
                if s0[i] in x:post_out[i] += 0.8
                else:
                    for x0 in list(jieba.cut(s0[i])):
                        if x0 in x:post_out[i] += 0.2
            max = post_out.max()
            max = 1 if max==0 else max
            return post_out/max

        text_data0 = data.get("history_of_present_illness")
        GXY =  "血压" in text_data0

        text_data0 = present_cut(text_data0)
        text_data1 = past_cut(data.get("past_history"))
        text_data2 = data.get("supplementary_examination")
        if text_data2 == None:text_data2 = "暂缺"   #零填
        else:text_data2 = supplementary_cut(text_data2)
        text_data = text_data0+text_data1+text_data2 +"。".join(re.findall('[\u4e00-\u9fa5]+',data.get("chief_complaint")) )+"。".join(re.findall('[\u4e00-\u9fa5]+',data.get("physical_examination")) )


        text, mask = self._get_token( "".join(s0)+text_data, 512)
        output = self.model(text.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE))
        post_out = post_process(text_data0+text_data1+text_data2)  #匹配的项概率均+0.2
        output += post_out.to(DEVICE)
        if GXY:output[:,12] = 1
        result = {emr_id: output}
        return result

    def _postprocess(self, data):
        infer_output = None
        for k, v in data.items():
            pred_labels = v.cpu().detach().numpy()
            pred_labels = [1 if pred > 0.336 else 0 for pred in pred_labels[0]]
            pred_diagnosis = self._get_diagnosis(pred_labels)
            infer_output = {k: pred_diagnosis}
        return infer_output   #结果形式需要不变

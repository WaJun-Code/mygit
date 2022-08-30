import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

#可以把52个label过bert做embedding，最后cosine一下 
#加R-Drop和相似度矩阵mse

#with open("labels0.txt", 'r', encoding='utf8') as f:s0 = f.readlines()
#s0 = [len(i.strip()) for i in s0]   #各个对应长度
s0 = [2, 2, 2, 2, 4, 4, 3, 3, 5, 2, 2, 4, 3, 5, 3, 4, 2, 2, 2, 4, 3, 2, 3, 5, 2, 4, 2, 8, 3, 2, 6, 2, 4, 6, 1, 2, 5, 2, 4, 5, 5, 2, 2, 2, 4, 6, 3, 3, 2, 9, 2, 3]

class BaselineModel(nn.Module):
    def __init__(self, bert_path):
        super(BaselineModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 1)

    def forward(self, text, mask,offline=False):
        bert_out = self.bert(text, attention_mask=mask)[0]
        if offline==True:bert_out1 = self.bert(text, attention_mask=mask)[0]
        num = 0
        for i in s0:
            if num==0:
                label_tensor = bert_out[:,1+num:1+num+i,:].mean(dim=1).unsqueeze(dim=1)
                if offline==True:label_tensor1 = bert_out1[:,1+num:1+num+i,:].mean(dim=1).unsqueeze(dim=1)
            else:
                label_tensor = torch.cat((label_tensor, bert_out[:,1+num:1+num+i,:].mean(dim=1).unsqueeze(dim=1) ),dim=1)
                if offline==True:label_tensor1 = torch.cat((label_tensor1, bert_out1[:,1+num:1+num+i,:].mean(dim=1).unsqueeze(dim=1) ),dim=1)
            num += i
        bert_out = bert_out[:,len(s0)+1:,:].mean(dim=1)
        if offline==True:
            bert_out1 = bert_out1[:,len(s0)+1:,:].mean(dim=1)
            RDrop_loss = F.kl_div(F.log_softmax(bert_out,dim=1),F.softmax(bert_out1,dim=1),reduction='mean')+F.kl_div(F.log_softmax(bert_out1,dim=1),F.softmax(bert_out,dim=1),reduction='mean')
            #再加一个相似度矩阵loss
            L_sim = torch.sqrt((label_tensor**2).sum(dim=-1)).unsqueeze(dim=-1)
            L_sim = torch.matmul(label_tensor,label_tensor.permute(0,2,1))/torch.matmul( L_sim,L_sim.permute(0,2,1))
            L_sim1 = torch.sqrt((label_tensor1**2).sum(dim=-1)).unsqueeze(dim=-1)
            L_sim1 = torch.matmul(label_tensor1,label_tensor1.permute(0,2,1))/torch.matmul( L_sim1,L_sim1.permute(0,2,1))
            sim_loss = ((L_sim-L_sim1)**2).mean()
        
        # out_linear = torch.cosine_similarity(label_tensor,bert_out.unsqueeze(dim=1).repeat(1,len(s0),1),dim=-1)

        bert_out = label_tensor*bert_out.unsqueeze(dim=1).repeat(1,len(s0),1)   #做点积实现交叉
        bert_out = self.dropout(bert_out)
        out_linear = self.fc(bert_out).squeeze(dim=-1)
        output = torch.sigmoid(out_linear)  #用52个label的embedding来做cosine

        if offline==True:return output,sim_loss+RDrop_loss
        else:return output
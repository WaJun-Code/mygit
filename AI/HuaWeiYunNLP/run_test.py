import logging
import torch
from mol_main import mol_features, get_protein_vec
from model import *
import pandas as pd
import time
import tqdm
start_time=time.time()

LOGGER = logging.getLogger(__file__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

protein_dim = 100
atom_dim = 34
hid_dim = 64
n_layers = 6
n_heads = 8
pf_dim = 512
dropout = 0.2
batch = 8
lr = 1e-4
lr_decay = 1.0
weight_decay = 1e-4
decay_interval = 5
iteration = 1
kernel_size = 7

encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
model0 = Predictor(encoder, decoder, device)
model0.load_state_dict(torch.load('TransformerCPI_model.pkl', map_location='cpu'))
model0.eval()   #freeze所有参数，必要
model0.to(device)

@torch.no_grad()
def inference(molecule_smiles: str, protein_fasta: str) -> float:
    LOGGER.debug(f'inference: smiles={molecule_smiles}, fasta={protein_fasta}')
    prob = float('nan')
    try:
        atom_feature, adj = mol_features(molecule_smiles)
        proteins = get_protein_vec(protein_fasta)
        labels = torch.tensor([0])
        data_pack = pack([atom_feature.to(device)], [adj.to(device)], [proteins.to(device)], [labels.to(device)], device)
        correct_labels, predicted_labels, prob = model0(data_pack, train=False)
        prob = prob.item()
    except Exception as e:
        LOGGER.error(f'inference - failed: smiles={molecule_smiles}, fasta={protein_fasta}, error={e}')
    else:
        LOGGER.info(f'inference - success: smiles={molecule_smiles}, fasta={protein_fasta}, prob={prob}')
    return prob
def gather_frac(T, S):   #对应线上指标
    T, S=np.array(T), np.array(S)
    indexS=np.argsort(S)
    k=0
    for i in range(1,1001):
        if T[indexS[-i]]==1:k+=1.2204
    return k
df=pd.read_csv('../data/val_data.csv',header=None)
T=df.iloc[: , 2].tolist()
S,num=[],0
for i in tqdm.tqdm(range(df.shape[0])):
    smiles, protein, label = df.iloc[i,0], df.iloc[i,1], df.iloc[i,2]
    S.append(inference(smiles, protein))
    if S[i]>0.5:num+=1
print("预测为正的样本数(个):" ,num)
AUC = roc_auc_score(T, S)
#PRC = precision_score(T, Y)
frac=gather_frac(T, S)
print(AUC, frac)
print("预测耗时：",(time.time()-start_time)/60 )
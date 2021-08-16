import os
import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from model import *
import time,timeit
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from gensim.models import Word2Vec
W2V_MODEL = Word2Vec.load("word2vec_30.model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results
def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])
def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return torch.tensor(atom_feat, dtype=torch.float), torch.tensor(adj_matrix, dtype=torch.float)
def get_protein_vec(protein, k=3):  # length: 200
    try:
        vectors = torch.tensor([list(W2V_MODEL.wv[protein[i:i+k]]) for i in range(len(protein) - k + 1)], dtype=torch.float)
    except:
        print("有word2vec无法识别")
        vectors = []
        for i in range(len(protein) - k + 1):
            try:
                vectors.append(list(W2V_MODEL.wv[protein[i:i+k]]))
            except:
                vectors.append(np.zeros([100,]).tolist())
        vectors = torch.tensor(vectors, dtype=torch.float)
    return vectors
class CPIDataset(Dataset):
    def __init__(self, file, limit=None):
        self.samples = []
        with open(file, 'r') as fd:  # Row Format: SMILES Protein Label
            for line in tqdm.tqdm(fd.readlines(), total=limit,
                                  desc=f'Loading dataset from "{os.path.basename(file)}"'):
                smiles, protein, label = line.strip().split(',')
                atom_feature, adj = mol_features(smiles)
                self.samples.append((
                    atom_feature, adj,
                    get_protein_vec(protein),
                    torch.tensor([float(label)])
                ))
                if limit is not None and len(self.samples) >= limit:
                    break
    @staticmethod
    def collate(samples):
        return map(torch.stack, zip(*samples))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=CPIDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

if __name__ == '__main__':
    epochs,batchsize,limits=10,1,10000  # small-187495，总763604,后51200个做val,前712404

    dataset_dev = CPIDataset('../data/val_data.csv', limit=60060)
    # dataset_train = dataset_train.create_data_loader(batch_size=batchsize, shuffle=True, drop_last=True)
    # dataset_dev = dataset_dev.create_data_loader(batch_size=batchsize, shuffle=True, drop_last=True)

    """ create model ,trainer and tester """
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
    iteration = 21
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    #model.load_state_dict(torch.load('TransformerCPI_model.pkl'))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=7,n_layer=3,batch=64'+ '.txt'
    file_model ='TransformerCPI_model.pkl'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tgather_frac')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_gather_frac = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train('../data/train_data.csv', limits, device)
        AUC_dev, PRC_dev, gather_frac = tester.test(dataset_dev)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,PRC_dev, gather_frac]
        tester.save_AUCs(AUCs, file_AUCs)
        if gather_frac > max_gather_frac:
            tester.save_model(model, 'TransformerCPI_model9.pkl')
            max_gather_frac = gather_frac
        tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
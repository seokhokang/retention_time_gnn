import numpy as np
import os
import pandas as pd
import pickle as pkl
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')

def preprocess(mol_list, y_list, name):

    def add_mol(mol_dict, mol, label):
    
        def _DA(mol):
    
            D_list, A_list = [], []
            for feat in chem_feature_factory.GetFeaturesForMol(mol):
                if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
            
            return D_list, A_list
    
        def _chirality(atom):
    
            if atom.HasProp('Chirality'):
                assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
            else:
                c_list = [0, 0]
    
            return c_list
    
        def _stereochemistry(bond):
    
            if bond.HasProp('Stereochemistry'):
                assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
            else:
                s_list = [0, 0]
    
            return s_list    
            
        n_node = mol.GetNumAtoms()
        n_edge = mol.GetNumBonds() * 2
        
        D_list, A_list = _DA(mol)
        rings = mol.GetRingInfo().AtomRings()
        atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
        atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1]
        atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
        atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
        atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
        atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
        atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
        atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
        
        node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)
    
        mol_dict['n_node'].append(n_node)
        mol_dict['n_edge'].append(n_edge)
        mol_dict['node_attr'].append(node_attr)
        
        if n_edge > 0:
    
            bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
            bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
            bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
            
            edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
            edge_attr = np.vstack([edge_attr, edge_attr])
            
            bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
            src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
            dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
            
            mol_dict['edge_attr'].append(edge_attr)
            mol_dict['src'].append(src)
            mol_dict['dst'].append(dst)
    
        mol_dict['label'].append(label)
        
        return mol_dict
    
    atom_list = ['H','C','N','O','F','Si','P','S','Cl','Br','I']
    charge_list = [1,-1, 0]
    degree_list = [1, 2, 3, 4, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','S']
    hydrogen_list = [1, 2, 3, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    
    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'label': []}
                     
    for i, mol in enumerate(mol_list):
    
        try:
            si = Chem.FindPotentialStereo(mol)
            for element in si:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                    mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                    mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
            assert '.' not in Chem.MolToSmiles(mol)
        except:
            print('exception', i)
            continue
    
        mol = Chem.RemoveHs(mol)
        mol_dict = add_mol(mol_dict, mol, y_list[i])
    
        if (i+1) % 10000 == 0: print('--- %d/%d processed' %(i+1, len(mol_list)))
    
    print('--- %d/%d processed' %(i+1, len(mol_list)))   
    
    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    mol_dict['label'] = np.hstack(mol_dict['label']).astype(float)

    np.savez_compressed('./dataset_graph_%s.npz'%name, data = [mol_dict])
    
    print('--- %s, %d mols'%(name, len(mol_dict['label'])))
    print('')



if __name__ == "__main__":


    ## METLIN-SMRT
    ## The dataset can be downloaded from
    ## https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913
    suppl = Chem.SDMolSupplier('./SMRT_dataset.sdf', removeHs = False) 
    
    mol_list = []
    RT_list = []
    for i, mol in enumerate(suppl):
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
    
        RT = mol.GetDoubleProp('RETENTION_TIME')
        if RT <= 200: continue
    
        mol_list.append(mol)
        RT_list.append(RT)
    
    print('METLIN-SMRT', len(RT_list), len(mol_list), np.min(RT_list), np.max(RT_list))
    preprocess(mol_list, RT_list, 'SMRT')

    
    ## Target Datasets from PredRet Database
    ## The dataset can be downloaded from
    ## http://predret.org/
    file_list = os.listdir('./predret/')
    
    for f in file_list:
        target = f.split('.')[0]
        data = pd.read_csv('./predret/%s.csv'%target)
        
        mol_list = [Chem.inchi.MolFromInchi(inchi, removeHs = False) for inchi in data['InChI'].to_numpy()]
        RT_list = data['RT'].to_numpy()
        
        print(target, len(RT_list), len(mol_list), np.min(RT_list), np.max(RT_list))
        preprocess(mol_list, RT_list, target)


    ## Target Datasets from MoNA Database
    ## The dataset can be downloaded from
    ## https://mona.fiehnlab.ucdavis.edu/
    file_list = os.listdir('./mona/')
    
    for f in file_list:
        target = f.split('.')[0]
        data = pd.read_csv('./mona/%s.csv'%target)
        
        mol_list = [Chem.inchi.MolFromInchi(inchi, removeHs = False) for inchi in data['InChI'].to_numpy()]
        RT_list = data['RT'].to_numpy()
        
        print(target, len(RT_list), len(mol_list), np.min(RT_list), np.max(RT_list))
        preprocess(mol_list, RT_list, target)

import os
import sys
import multiprocessing
from functools import partial
import pickle
import torch
import dgl
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, rdchem
from Bio.PDB import PDBParser, Selection, PDBIO, Select
from scipy.spatial import distance_matrix

# Add current dir to path
sys.path.append(os.getcwd())

BT = rdchem.BondType
allowable_features = {
    'possible_chirality_list': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True]
}
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'SEC': 'C', 'MSE': 'M'}
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']; pro_res_aromatic_table = ['F', 'W', 'Y']; pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']; pro_res_acidic_charged_table = ['D', 'E']; pro_res_basic_charged_table = ['H', 'K', 'R']
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def safe_index(l, e): 
    try: return l.index(e)
    except: return len(l)-1

def prop(residue):
    return [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0, 1 if residue in pro_res_polar_neutral_table else 0, 1 if residue in pro_res_acidic_charged_table else 0, 1 if residue in pro_res_basic_charged_table else 0, 1]

def res_pocket(p):
    parser = PDBParser(PERMISSIVE=1)
    s = parser.get_structure('a', p)
    atom2res_idx, res_name, count = [], [], -1
    for model in s:
        for chain in model: 
            for residue in chain:
                count += 1; name = residue.get_resname()
                for atom in residue: 
                    if atom.get_fullname()[0] != 'H' and atom.get_fullname()[1] != 'H':
                        atom2res_idx.append(count); res_name.append(name)
    res3 = [three_to_one[i] if i in three_to_one.keys() else 'X' for i in res_name]
    return atom2res_idx, [pro_res_table.index(i) for i in res3], [prop(i) for i in res3]

def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    return [[safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())), safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()), safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()), safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()), safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())), allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()), safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()), safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()), safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx))] for idx, atom in enumerate(mol.GetAtoms())]

def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: return [[0]]
    cliques = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0: return [list(range(n_atoms))]
    for bond in res:
        if [bond[0][0], bond[0][1]] in cliques: cliques.remove([bond[0][0], bond[0][1]])
        else: cliques.remove([bond[0][1], bond[0][0]])
        cliques.append([bond[0][0]]); cliques.append([bond[0][1]])
    for c in range(len(cliques) - 1):
        for k in range(c + 1, len(cliques)):
            if set(cliques[c]) & set(cliques[k]):
                cliques[c] = list(set(cliques[c]) | set(cliques[k])); cliques[k] = []
    return [c for c in cliques if c]

def group_complex(ligand, pocket_dir):
    brics_index = brics_decomp(ligand)
    group_index = [0 for _ in range(len(ligand.GetAtoms()))]
    group_type = [0 for _ in range(len(ligand.GetAtoms()))]
    group_prop = [[0,0,0,0,0,0] for _ in range(len(ligand.GetAtoms()))]
    for i, idx in enumerate(brics_index):
        for idx_ in idx: group_index[idx_] = i
    atom2res_idx, res_type, res_prop = res_pocket(pocket_dir)
    group_index.extend([i+len(brics_index) for i in atom2res_idx])
    group_type.extend([i+1 for i in res_type])
    group_prop.extend(res_prop)
    return torch.tensor(group_index, dtype=torch.float32), torch.tensor(group_type, dtype=torch.float32), torch.tensor(group_prop, dtype=torch.float32)

def bond_featurizer(ligand, pocket, index):
    bond_type = []; num_atoms = len(ligand.GetAtoms())
    for a1, a2 in zip(index[0].tolist(), index[1].tolist()):
        if a1 < num_atoms and a2 < num_atoms: bond = ligand.GetBondBetweenAtoms(int(a1), int(a2)); bond_type.append(bonds[bond.GetBondType()] if bond else 4)
        elif a1 >= num_atoms and a2 >= num_atoms: bond = pocket.GetBondBetweenAtoms(int(a1-num_atoms), int(a2-num_atoms)); bond_type.append(bonds[bond.GetBondType()] if bond else 4)
        else: bond_type.append(4)
    return torch.tensor(bond_type, dtype=torch.float32)

def Graph_Information(ligand_file, pocket_file):
    ligand = Chem.RemoveAllHs(Chem.MolFromMolFile(ligand_file))
    pocket = Chem.RemoveAllHs(Chem.MolFromPDBFile(pocket_file))
    if not pocket or not ligand: raise ValueError("Could not load pocket or ligand. Check file sanity.")
    G = dgl.heterograph({('atom', 'int', 'atom'): ([], []), ('atom', 'ind', 'atom'): ([], [])})
    l_at, p_at = np.array([i.GetAtomicNum() for i in ligand.GetAtoms()]), np.array([i.GetAtomicNum() for i in pocket.GetAtoms()])
    graph_x = torch.tensor(np.concatenate([l_at, p_at]))
    G.add_nodes(len(l_at) + len(p_at))
    coor_l, coor_p = ligand.GetConformer().GetPositions(), pocket.GetConformer().GetPositions()
    pos = torch.tensor(np.concatenate([coor_l, coor_p]), dtype=torch.float32)
    for i in range(len(coor_l)):
        for j in range(i + 1, len(coor_l)):
            if 0 < np.linalg.norm(coor_l[i] - coor_l[j]) <= 5: G.add_edges([i, j], [j, i], etype='int')
    for b in pocket.GetBonds():
        u, v = b.GetBeginAtomIdx() + len(coor_l), b.GetEndAtomIdx() + len(coor_l); G.add_edges([u, v], [v, u], etype='int')
    for i in range(len(coor_l)):
        for j in range(len(coor_p)):
            if 0 < np.linalg.norm(coor_l[i] - coor_p[j]) <= 5: G.add_edges([i, len(coor_l) + j], [len(coor_l) + j, i], etype='int')
    conn_p = set(int(j) for i in range(len(coor_l)) for j in G.successors(i, etype='int') if j >= len(coor_l))
    for i in conn_p:
        for j in range(len(coor_p)):
            if i != len(coor_l)+j and 0 < np.linalg.norm(coor_p[i-len(coor_l)] - coor_p[j]) <= 3 and not G.has_edges_between(i, len(coor_l)+j, etype='int'): G.add_edges([i, len(coor_l)+j], [len(coor_l)+j, i], etype='int')
    index, g_type, g_prop = group_complex(ligand, pocket_file)
    G.nodes['atom'].data.update({'x': graph_x, 'pos': pos, 'type': torch.tensor(np.concatenate([[1]*len(l_at), [0]*len(p_at)])), 'atom_scalar': torch.tensor(np.concatenate([lig_atom_featurizer(ligand), lig_atom_featurizer(pocket)]), dtype=torch.float32), 'res_idx': index, 'res_type': g_type, 'res_prop': g_prop})
    G.edges['int'].data['bond_scalar'] = bond_featurizer(ligand, pocket, G.edges(etype='int'))
    G.edges['ind'].data['bond_scalar'] = bond_featurizer(ligand, pocket, G.edges(etype='ind'))
    return G

def graph_save(lig_f, pock_f, save_f):
    if not os.path.exists(lig_f) or not os.path.exists(pock_f): return
    try:
        temp_f = save_f + '.tmp'
        with open(temp_f, 'wb') as f: 
            pickle.dump(Graph_Information(lig_f, pock_f), f)
        os.replace(temp_f, save_f)
    except Exception as e:
        if os.path.exists(save_f + '.tmp'): os.remove(save_f + '.tmp')
        print(f"Error with {os.path.basename(lig_f)}: {e}")

def parallel_graph_save(sdf_name, ligand_dir, pocket_f, pickle_dir):
    lig_f = os.path.join(ligand_dir, sdf_name)
    save_f = os.path.join(pickle_dir, sdf_name.replace('.sdf', '.pkl'))
    if not os.path.exists(save_f) or os.path.getsize(save_f) == 0:
        graph_save(lig_f, pocket_f, save_f)
    return sdf_name

if __name__ == '__main__':
    dir_ = r'd:\PBCNet2.0\case\proQ'
    ligand_dir = os.path.join(dir_, 'ligands')
    pocket_f = os.path.join(dir_, 'pocket.pdb')
    pickle_dir = os.path.join(dir_, 'pickles')
    os.makedirs(pickle_dir, exist_ok=True)

    sdfs = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf')]
    print(f"Starting Multi-core Processing for {len(sdfs)} ligands...")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    func = partial(parallel_graph_save, ligand_dir=ligand_dir, pocket_f=pocket_f, pickle_dir=pickle_dir)
    for i, sdf_name in enumerate(pool.imap_unordered(func, sdfs), 1):
        print(f"[{i}/{len(sdfs)}] Finished: {sdf_name}")

    pool.close()
    pool.join()
    print("Pre-processing Complete! All graphs generated.")

import json
import os

notebook_path = r'd:\PBCNet2.0\case\try_proQ.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Pocket Extraction to use UNIQUE names
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def pocket_extract' in ''.join(cell['source']):
        cell['source'] = [
            "def extract(ligand, pdb_path, sdf_path):\n",
            "    if not ligand: return False\n",
            "    parser = PDBParser(PERMISSIVE=1)\n",
            "    structure = parser.get_structure(\"protein\", pdb_path)\n",
            "    lp = []\n",
            "    for l in ligand:\n",
            "        try: lp.append(l.GetConformer().GetPositions())\n",
            "        except: continue\n",
            "    if not lp: return False\n",
            "    ligand_positions = np.concatenate(lp)\n",
            "    class ResidueSelect(Select):\n",
            "        def accept_residue(self, residue):\n",
            "            residue_positions = np.array([np.array(list(atom.get_vector())) for atom in residue.get_atoms() if \"H\" not in atom.get_id()])\n",
            "            if len(residue_positions.shape) < 2: return 0\n",
            "            return 1 if np.min(distance_matrix(residue_positions, ligand_positions)) < 8.0 else 0\n",
            "    io = PDBIO(); io.set_structure(structure)\n",
            "    # Save UNIQUE pocket for each ligand\n",
            "    pocket_path = sdf_path.replace('ligands', 'pockets').replace('.sdf', '_pocket.pdb')\n",
            "    os.makedirs(os.path.dirname(pocket_path), exist_ok=True)\n",
            "    io.save(pocket_path, ResidueSelect())\n",
            "    return True\n",
            "\n",
            "def pocket_extract(sdf_files, protein_file):\n",
            "    for a in sdf_files:\n",
            "        mol = Chem.MolFromMolFile(a)\n",
            "        if mol:\n",
            "            if extract([mol], protein_file, a):\n",
            "                pass # print(f\"Extracted: {os.path.basename(a)}\")\n",
            "            else:\n",
            "                print(f\"Skipping: No conformer in {os.path.basename(a)}\")"
        ]

# 2. Update Graph Generation to be ROBUST
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'graph_save(lig_f, pock_f, save_f)' in ''.join(cell['source']):
        cell['source'] = [
            "def graph_save(lig_f, pock_f, save_f):\n",
            "    if not os.path.exists(lig_f) or not os.path.exists(pock_f): return\n",
            "    try:\n",
            "        with open(save_f, 'wb') as f: \n",
            "            pickle.dump(Graph_Information(lig_f, pock_f), f)\n",
            "    except Exception as e:\n",
            "        print(f\"Error generating graph for {os.path.basename(lig_f)}: {e}\")"
        ]

# 3. Update the execution loop for Graph Generation
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'data_list = [[' in ''.join(cell['source']):
        cell['source'] = [
            "print(\"Starting Graph Generation...\")\n",
            "count = 0\n",
            "sdfs = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf')]\n",
            "for i in sdfs:\n",
            "    lig_f = os.path.join(ligand_dir, i)\n",
            "    pock_f = lig_f.replace('ligands', 'pockets').replace('.sdf', '_pocket.pdb')\n",
            "    save_f = lig_f.replace('ligands', 'pickles').replace('.sdf', '.pkl')\n",
            "    os.makedirs(os.path.dirname(save_f), exist_ok=True)\n",
            "    \n",
            "    if not os.path.exists(save_f):\n",
            "        graph_save(lig_f, pock_f, save_f)\n",
            "    \n",
            "    count += 1\n",
            "    if count % 100 == 0: print(f\"Processed {count}/{len(sdfs)} graphs...\")\n",
            "print(\"Graph Generation Complete!\")"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched successfully with Unique Pockets and Robust Loops!")

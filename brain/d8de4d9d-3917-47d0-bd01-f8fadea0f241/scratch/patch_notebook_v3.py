import json
import os

notebook_path = r'd:\PBCNet2.0\case\try_proQ.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Ensure Graph Generation loop is ROBUST (Single Pocket)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'graph_save(lig_f, pock_f, save_f)' in ''.join(cell['source']):
        cell['source'] = [
            "def graph_save(lig_f, pock_f, save_f):\n",
            "    if not os.path.exists(lig_f) or not os.path.exists(pock_f): return\n",
            "    try:\n",
            "        with open(save_f, 'wb') as f: \n",
            "            pickle.dump(Graph_Information(lig_f, pock_f), f)\n",
            "    except Exception as e:\n",
            "        print(f\"Error with {os.path.basename(lig_f)}: {e}\")"
        ]

# 2. Fix the loop to keep going even if one file fails
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'data_list = [[' in ''.join(cell['source']):
        cell['source'] = [
            "pocket_f = os.path.join(dir_, 'pocket.pdb')\n",
            "pickle_dir = os.path.join(dir_, 'pickles')\n",
            "os.makedirs(pickle_dir, exist_ok=True)\n",
            "\n",
            "sdfs = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf')]\n",
            "print(f\"Processing {len(sdfs)} ligands...\")\n",
            "\n",
            "for i, sdf_name in enumerate(sdfs):\n",
            "    lig_f = os.path.join(ligand_dir, sdf_name)\n",
            "    save_f = os.path.join(pickle_dir, sdf_name.replace('.sdf', '.pkl'))\n",
            "    \n",
            "    if not os.path.exists(save_f):\n",
            "        graph_save(lig_f, pocket_f, save_f)\n",
            "    \n",
            "    if (i+1) % 100 == 0: print(f\"Done {i+1}/{len(sdfs)}...\")\n",
            "print(\"Graph Generation Finished!\")"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched! One pocket.pdb, but with a robust loop.")

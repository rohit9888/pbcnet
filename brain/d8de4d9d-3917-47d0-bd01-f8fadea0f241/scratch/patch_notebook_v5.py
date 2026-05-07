import json
import os

notebook_path = r'd:\PBCNet2.0\case\try_proQ.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Graph Generation logic (v5 - Final & Robust)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'graph_save(lig_f, pock_f, save_f)' in ''.join(cell['source']):
        cell['source'] = [
            "def graph_save(lig_f, pock_f, save_f):\n",
            "    if not os.path.exists(lig_f) or not os.path.exists(pock_f): return\n",
            "    try:\n",
            "        # Use a temp file and rename it to avoid 0-byte corruption if interrupted\n",
            "        temp_f = save_f + '.tmp'\n",
            "        with open(temp_f, 'wb') as f: \n",
            "            pickle.dump(Graph_Information(lig_f, pock_f), f)\n",
            "        os.replace(temp_f, save_f)\n",
            "    except Exception as e:\n",
            "        if os.path.exists(save_f + '.tmp'): os.remove(save_f + '.tmp')\n",
            "        print(f\"Error with {os.path.basename(lig_f)}: {e}\")"
        ]

# 2. Update the execution loop to use Multiprocessing & Correct Paths
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'multiprocessing' in ''.join(cell['source']):
        cell['source'] = [
            "import multiprocessing\n",
            "from functools import partial\n",
            "\n",
            "pocket_f = os.path.join(dir_, 'pocket.pdb')\n",
            "pickle_dir = os.path.join(dir_, 'pickles')\n",
            "os.makedirs(pickle_dir, exist_ok=True)\n",
            "\n",
            "def parallel_graph_save(sdf_name, ligand_dir, pocket_f, pickle_dir):\n",
            "    lig_f = os.path.join(ligand_dir, sdf_name)\n",
            "    save_f = os.path.join(pickle_dir, sdf_name.replace('.sdf', '.pkl'))\n",
            "    # Only skip if the file exists AND is valid (not 0 bytes)\n",
            "    if not os.path.exists(save_f) or os.path.getsize(save_f) == 0:\n",
            "        graph_save(lig_f, pocket_f, save_f)\n",
            "\n",
            "sdfs = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf')]\n",
            "print(f\"Resuming Graph Generation for {len(sdfs)} ligands...\")\n",
            "\n",
            "pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())\n",
            "func = partial(parallel_graph_save, ligand_dir=ligand_dir, pocket_f=pocket_f, pickle_dir=pickle_dir)\n",
            "\n",
            "for i, _ in enumerate(pool.imap_unordered(func, sdfs), 1):\n",
            "    if i % 100 == 0: print(f\"Processed {i}/{len(sdfs)} graphs...\")\n",
            "\n",
            "pool.close()\n",
            "pool.join()\n",
            "print(\"All Graphs Generated Successfully!\")"
        ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched (v5)! Correct paths, multiprocessing, and error-handling active.")

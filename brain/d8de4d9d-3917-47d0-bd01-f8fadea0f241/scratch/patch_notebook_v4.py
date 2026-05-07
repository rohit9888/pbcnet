import json
import os

notebook_path = r'd:\PBCNet2.0\case\try_proQ.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the Graph Generation loop to use Multiprocessing
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'graph_save(lig_f, pocket_f, save_f)' in ''.join(cell['source']):
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
            "    if not os.path.exists(save_f):\n",
            "        graph_save(lig_f, pocket_f, save_f)\n",
            "\n",
            "sdfs = [i for i in os.listdir(ligand_dir) if i.endswith('.sdf')]\n",
            "print(f\"Starting Parallel Graph Generation for {len(sdfs)} ligands...\")\n",
            "\n",
            "pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())\n",
            "func = partial(parallel_graph_save, ligand_dir=ligand_dir, pocket_f=pocket_f, pickle_dir=pickle_dir)\n",
            "\n",
            "for i, _ in enumerate(pool.imap_unordered(func, sdfs), 1):\n",
            "    if i % 100 == 0: print(f\"Done {i}/{len(sdfs)}...\")\n",
            "\n",
            "pool.close()\n",
            "pool.join()\n",
            "print(\"Parallel Graph Generation Finished!\")"
        ]
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched with Multiprocessing! It will be much faster now.")

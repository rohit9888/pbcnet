import json
import os

notebook_path = r'd:\PBCNet2.0\case\try_proQ.ipynb'

if not os.path.exists(notebook_path):
    print(f"Error: Could not find {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell containing pocket_extract and patch it
patched = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def pocket_extract' in ''.join(cell['source']):
        new_source = [
            'def extract(ligand, pdb_path):\n',
            '    if not ligand:\n',
            '        return False\n',
            '    parser = PDBParser(PERMISSIVE=1)\n',
            '    structure = parser.get_structure("protein", pdb_path)\n',
            '    \n',
            '    lp = []\n',
            '    for l in ligand:\n',
            '        try:\n',
            '            lp.append(l.GetConformer().GetPositions())\n',
            '        except:\n',
            '            continue\n',
            '    \n',
            '    if not lp:\n',
            '        return False\n',
            '        \n',
            '    ligand_positions = np.concatenate(lp)\n',
            '\n',
            '    class ResidueSelect(Select):\n',
            '        def accept_residue(self, residue):\n',
            '            residue_positions = np.array([np.array(list(atom.get_vector())) for atom in residue.get_atoms() if "H" not in atom.get_id()])\n',
            '            if len(residue_positions.shape) < 2: return 0\n',
            '            min_dis = np.min(distance_matrix(residue_positions, ligand_positions))\n',
            '            return 1 if min_dis < 8.0 else 0\n',
            '\n',
            '    io = PDBIO()\n',
            '    io.set_structure(structure)\n',
            '    pocket_path = pdb_path.replace(\'protein.pdb\', \'pocket.pdb\')\n',
            '    io.save(pocket_path, ResidueSelect())\n',
            '    return True\n',
            '\n',
            'def pocket_extract(sdf_files, protein_file):\n',
            '    ligands = []\n',
            '    for a in sdf_files:\n',
            '        mol = Chem.MolFromMolFile(a)\n',
            '        if mol: ligands.append(mol)\n',
            '    \n',
            '    if extract(ligands, protein_file):\n',
            '        print("Pocket extracted successfully.")\n',
            '    else:\n',
            '        print(f"Skipping: No valid conformers found in {sdf_files}")'
        ]
        cell['source'] = new_source
        patched = True
        break

if patched:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully!")
else:
    print("Error: Could not find the pocket_extract function in the notebook.")

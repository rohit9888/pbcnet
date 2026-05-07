import os
from rdkit import Chem

# 1. Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_ = os.path.join(script_dir, 'proQ')
ligand_dir = os.path.join(dir_, 'ligands')
sdf_file = os.path.join(dir_, 'structures.sdf')

# 2. Create the ligands folder if it doesn't exist
if not os.path.exists(ligand_dir):
    os.makedirs(ligand_dir)
    print(f"Created directory: {ligand_dir}")

# 3. Use RDKit to safely split the SDF
print(f"Reading {sdf_file} using RDKit...")
try:
    # Use SDMolSupplier to read molecules one by one
    # sanitize=False helps if there are some minor chemistry errors in the file
    supplier = Chem.SDMolSupplier(sdf_file, sanitize=True, removeHs=False)
    
    count = 0
    for mol in supplier:
        if mol is not None:
            filename = os.path.join(ligand_dir, f"ligand_{count}.sdf")
            writer = Chem.SDWriter(filename)
            writer.write(mol)
            writer.close()
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} ligands...")
        else:
            print(f"Warning: Could not read molecule at index {count + 1}")

    print(f"Successfully split {count} ligands into {ligand_dir}")
except Exception as e:
    print(f"An error occurred: {e}")

import os
from ase.io import read, write
import numpy as np

# Input path
INPUT_VASP_FILE = "./vasprun.xml" 

# Output path
TRAIN_OUTPUT = "./ymno3_train.xyz"
VAL_OUTPUT = "./ymno3_val.xyz"
TEST_OUTPUT = "./ymno3_test.xyz"

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Remaining 0.1 is test

def process_data():
    print(f"Reading data from {INPUT_VASP_FILE}...")
    try:
        # index=':' reads the entire trajectory
        atoms_list = read(INPUT_VASP_FILE, index=":")
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Note: If you only have XDATCAR, you cannot fine-tune because it lacks Energy/Forces.")
        print("Please use vasprun.xml or OUTCAR from your VASP calculation.")
        return

    print(f"Loaded {len(atoms_list)} frames.")

    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(atoms_list)

    # Calculate split indices
    n_total = len(atoms_list)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_data = atoms_list[:n_train]
    val_data = atoms_list[n_train:n_train+n_val]
    test_data = atoms_list[n_train+n_val:]

    print(f"Splitting data:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")

    # Save to XYZ (ASE automatically saves info['energy'] and arrays['forces'] to extxyz)
    write(TRAIN_OUTPUT, train_data)
    write(VAL_OUTPUT, val_data)
    write(TEST_OUTPUT, test_data)

    print("Processing complete. Files saved.")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(".", exist_ok=True)
    process_data()
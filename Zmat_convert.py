import numpy as np
import pubchempy as pcp
from pyscf.lib import geometry
import os

# --- 1. THE CONVERTER LOGIC (From your previous script) ---

def cartesian_to_zmatrix(atoms, coords):
    """
    Converts Cartesian coordinates to a Z-matrix structure using PySCF.
    """
    # PySCF expects a list of [atom_symbol, x, y, z]
    pyscf_geom = []
    for atom, coord in zip(atoms, coords):
        pyscf_geom.append([atom] + list(coord))

    # Generate Z-matrix
    z_matrix_str, parameters = geometry.build_z_matrix(pyscf_geom)
    
    # Format Z-Matrix Lines
    z_matrix_lines = z_matrix_str.split('\n')
    
    # Format Variables
    variables_dict = {}
    for key, value in parameters.items():
        variables_dict[key] = f'{value:.6f}' 

    return z_matrix_lines, variables_dict


# --- 2. THE MOLECULE BANK FETCHER ---

def fetch_molecule_data(name):
    """
    Searches PubChem for a molecule by name and retrieves its 3D structure.
    Returns (atoms, coords) or None if not found/no 3D data.
    """
    print(f"Searching PubChem for '{name}' (3D)...")
    
    # 'record_type=3d' is crucial! Standard PubChem records are often just 2D flat maps.
    compounds = pcp.get_compounds(name, 'name', record_type='3d')

    if not compounds:
        print(f"  [!] Could not find a 3D record for '{name}'. Skipping.")
        return None, None

    compound = compounds[0] # Take the first result
    
    # Extract atoms and coordinates
    atoms = []
    coords = []
    
    for atom in compound.atoms:
        atoms.append(atom.element)
        # PubChem usually gives coords in Angstroms
        coords.append([atom.x, atom.y, atom.z])
        
    return atoms, np.array(coords)


# --- 3. BATCH PROCESSOR ---

def process_molecule_bank(molecule_names, output_dir="gaussian_inputs"):
    """
    Iterates through a list of names, converts them, and saves .gjf files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- Starting Batch Processing of {len(molecule_names)} Molecules ---\n")

    for name in molecule_names:
        # A. Fetch Data
        atoms, coords = fetch_molecule_data(name)
        if atoms is None:
            continue

        # B. Convert to Z-Matrix
        try:
            zmat_lines, zmat_vars = cartesian_to_zmatrix(atoms, coords)
        except Exception as e:
            print(f"  [!] Error converting {name} to Z-Matrix: {e}")
            continue

        # C. Write Gaussian Input File
        filename = f"{name.replace(' ', '_')}.gjf"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(f"%Chk={name}.chk\n") # Checkpoint file
            f.write("# RHF/6-31G(d) Opt Freq\n\n")
            f.write(f"{name} optimized geometry from PubChem 3D\n\n")
            f.write("0 1\n") # Charge and Multiplicity (Assumed Neutral Singlet)
            
            # Write Geometry
            f.write('\n'.join(zmat_lines) + '\n\n')
            
            # Write Variables
            for var, val in zmat_vars.items():
                f.write(f"{var}={val}\n")
            
            f.write('\n') # Final blank line

        print(f"  [+] Successfully generated: {filepath}")

    print("\n--- Processing Complete ---")


# --- 4. EXECUTION ---

if __name__ == "__main__":
    # DEFINE YOUR MOLECULE BANK HERE
    my_molecules = [
        "Water",
        "Methane",
        "Ethanol",
        "Caffeine",
        "Benzene",
        "Aspirin"
    ]

    process_molecule_bank(my_molecules)
    
# This script converts Cartesian coordinates (XYZ format) into a Z-matrix
# structure suitable for Gaussian input files, leveraging the geometry tools
# provided by the scientific Python library 'pyscf'.

# You may need to install pyscf: pip install pyscf

import numpy as np
from pyscf.lib import geometry

def cartesian_to_zmatrix(atoms, coords):
    """
    Converts Cartesian coordinates to a Z-matrix structure.

    Args:
        atoms (list): List of atomic symbols (e.g., ['O', 'H', 'H']).
        coords (list of lists): Cartesian coordinates [[x1, y1, z1], ...].

    Returns:
        tuple: (z_matrix_lines, variables_dict) for Gaussian input.
    """
    if len(atoms) != len(coords):
        raise ValueError("Atom list and coordinate list must have the same length.")

    # Convert lists to PySCF's expected format (a list of [atom_symbol, x, y, z])
    pyscf_geom = []
    for atom, coord in zip(atoms, coords):
        pyscf_geom.append([atom] + list(coord))

    # PySCF's geometry.build_z_matrix handles the complex internal coordinate math.
    # It returns a tuple: (z_matrix_string, parameters_dict)
    # The string uses variables (R1, A1, D1), and the dict contains their values.
    # Note: 'build_z_matrix' requires the geometry to be slightly perturbed
    # if it is perfectly linear or planar, but often works fine for non-ideal inputs.
    z_matrix_str, parameters = geometry.build_z_matrix(pyscf_geom)

    # 1. Format the Z-Matrix Section (e.g., O, H 1 R1, H 1 R2 2 A1)
    z_matrix_lines = z_matrix_str.split('\n')
    
    # 2. Format the Variables Section (e.g., R1=0.96, A1=104.5)
    variables_dict = {}
    for key, value in parameters.items():
        # Format values to high precision scientific notation
        variables_dict[key] = f'{value:.6f}' 

    return z_matrix_lines, variables_dict


# --- 1. DEFINE YOUR INPUT COORDINATES (Example: Water Molecule) ---

# Atomic symbols
water_atoms = ['O', 'H', 'H']

# Cartesian Coordinates (in Angstroms)
water_coords = np.array([
    [ 0.000000,  0.000000, -0.124800],  # O
    [ 0.790695,  0.000000,  0.495200],  # H
    [-0.790695,  0.000000,  0.495200],  # H
])


# --- 2. PERFORM THE CONVERSION ---

try:
    zmat_lines, zmat_vars = cartesian_to_zmatrix(water_atoms, water_coords)
except Exception as e:
    print(f"An error occurred during Z-matrix generation. Ensure your molecule is not perfectly linear or planar, which can require advanced handling: {e}")
    exit()


# --- 3. FORMAT FOR GAUSSIAN INPUT (.com or .gjf file) ---

gaussian_input_file = "water_zmat.gjf"

print(f"--- Generating Gaussian Input File: {gaussian_input_file} ---\n")

gaussian_template = []
gaussian_template.append('# RHF/6-31G(d) Opt Freq')
gaussian_template.append('')
gaussian_template.append('Water Optimization via Z-Matrix')
gaussian_template.append('')
gaussian_template.append('0 1') # Charge and Multiplicity (Singlet Neutral)

# Append Z-Matrix definition lines
gaussian_template.extend(zmat_lines)
gaussian_template.append('')

# Append Variables section
for var, value in zmat_vars.items():
    gaussian_template.append(f'{var}={value}')

gaussian_template.append('\n') # Final newline is standard

# Display output to console
print('\n'.join(gaussian_template))


# --- 4. (Optional) Save to File ---
# with open(gaussian_input_file, 'w') as f:
#     f.write('\n'.join(gaussian_template))
# print(f"\nSuccessfully wrote Gaussian input to {gaussian_input_file}")

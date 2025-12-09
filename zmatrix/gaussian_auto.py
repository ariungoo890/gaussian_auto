import pubchempy as pcp
import numpy as np
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
OUTPUT_DIR = "gaussian_inputs"
METHOD = "RHF/6-31G(d)"
JOB_TYPE = "Opt Freq"      # Gaussian job type
COORD_FORMAT = "zmatrix"   # "cartesian" or "zmatrix"
# -------------------------------


# =============================================
#     GEOMETRY CONVERSION: CARTESIAN → ZMATRIX
# =============================================
def cartesian_to_zmatrix(atoms, coords):
    """
    Convert Cartesian coordinates to a Z-matrix.
    Returns a list of rows for the Z-matrix.
    """
    coords = np.array(coords)
    zmatrix = []

    def bond_length(i, j):
        return np.linalg.norm(coords[i] - coords[j])

    def bond_angle(i, j, k):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(cosang))

    def dihedral(i, j, k, l):
        p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[l]
        b0 = -1 * (p1 - p0)
        b1 = (p2 - p1)
        b2 = (p3 - p2)

        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))


    # Atom 1
    zmatrix.append([atoms[0]])

    # Atom 2
    if len(atoms) > 1:
        r = bond_length(1, 0)
        zmatrix.append([atoms[1], 1, r])

    # Atom 3
    if len(atoms) > 2:
        r = bond_length(2, 1)
        angle = bond_angle(2, 1, 0)
        zmatrix.append([atoms[2], 2, r, 1, angle])

    # Remaining atoms
    for i in range(3, len(atoms)):
        r = bond_length(i, i - 1)
        angle = bond_angle(i, i - 1, i - 2)
        dih = dihedral(i, i - 1, i - 2, i - 3)
        zmatrix.append([atoms[i], i, r, i - 1, angle, i - 2, dih])

    return zmatrix



# =============================================
#     FETCH 3D STRUCTURE FROM PUBCHEM
# =============================================
def fetch_3d_structure(name):
    """
    Fetches the 3D structure of a molecule from PubChem.
    Returns: (atoms_list, coordinates_list) or (None, None)
    """
    print(f"Searching PubChem for '{name}'...")

    try:
        compounds = pcp.get_compounds(name, 'name', record_type='3d')

        if not compounds:
            print(f"  [!] No 3D record found for '{name}'. Skipping.")
            return None, None

        compound = compounds[0]
        atoms = [atom.element for atom in compound.atoms]
        coords = [[atom.x, atom.y, atom.z] for atom in compound.atoms]

        return atoms, coords

    except Exception as e:
        print(f"  [!] Error fetching '{name}': {e}")
        return None, None



# =============================================
#     WRITE GAUSSIAN FILE
# =============================================
def write_gjf(name, atoms, coords, coord_format="cartesian"):
    """
    Writes a Gaussian .gjf file in either Cartesian or Z-matrix format.
    """

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    safe_name = name.replace(" ", "_")
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.gjf")

    with open(filepath, 'w') as f:

        # Link 0
        f.write(f"%Chk={safe_name}.chk\n")

        # Route section
        f.write(f"# {METHOD} {JOB_TYPE}\n\n")

        # Title
        f.write(f"{name} (Auto-generated)\n\n")

        # Charge and multiplicity
        f.write("0 1\n")

        # --- COORDINATES ---
        if coord_format.lower() == "cartesian":
            for atom, (x, y, z) in zip(atoms, coords):
                f.write(f"{atom:<2} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        elif coord_format.lower() == "zmatrix":
            zm = cartesian_to_zmatrix(atoms, coords)

            for row in zm:
                line = ""
                for item in row:
                    if isinstance(item, float):
                        line += f"  {item:10.6f}"
                    else:
                        line += f"  {item}"
                f.write(line.strip() + "\n")

        f.write("\n")

    print(f"  [+] Created: {filepath}")



# =============================================
#     PROCESS MOLECULE LIST
# =============================================
def process_bank(molecule_list):
    print(f"--- Processing {len(molecule_list)} Molecules ---\n")

    for mol in molecule_list:
        atoms, coords = fetch_3d_structure(mol)
        if atoms:
            write_gjf(mol, atoms, coords, coord_format=COORD_FORMAT)

    print(f"\n--- Batch Complete. Check the '{OUTPUT_DIR}' folder. ---")



# =============================================
#     MAIN
# =============================================
if __name__ == "__main__":
    bank = [
        "fructose",
        "glucose",
        "water",
        "caffeine"
    ]

    process_bank(bank)

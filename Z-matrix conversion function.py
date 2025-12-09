import numpy as np

def cartesian_to_zmatrix(atoms, coords):
    """
    Convert Cartesian coordinates to a Z-matrix.
    atoms: list of atomic symbols
    coords: Nx3 array of Cartesian coordinates
    """
    coords = np.array(coords)
    
    zmatrix = []

    def bond_length(i, j):
        return np.linalg.norm(coords[i] - coords[j])

    def bond_angle(i, j, k):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.degrees(np.arccos(cosang))

    def dihedral(i, j, k, l):
        p0, p1, p2, p3 = coords[i], coords[j], coords[k], coords[l]
        b0 = -1*(p1-p0)
        b1 = p2-p1
        b2 = p3-p2

        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

    # Atom 1
    zmatrix.append([atoms[0]])

    if len(atoms) > 1:
        # Atom 2
        r = bond_length(1, 0)
        zmatrix.append([atoms[1], 1, r])

    if len(atoms) > 2:
        # Atom 3
        r = bond_length(2, 1)
        angle = bond_angle(2, 1, 0)
        zmatrix.append([atoms[2], 2, r, 1, angle])

    # Remaining atoms
    for i in range(3, len(atoms)):
        r = bond_length(i, i-1)
        angle = bond_angle(i, i-1, i-2)
        dih = dihedral(i, i-1, i-2, i-3)
        zmatrix.append([atoms[i], i, r, i-1, angle, i-2, dih])

    return zmatrix

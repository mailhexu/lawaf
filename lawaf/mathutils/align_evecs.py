import numpy as np
from numpy.linalg import svd, eigh
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import copy


def rotation_matrix_3(theta1, theta2, theta3):
    R = Rotation.from_euler("xyz", [theta1, theta2, theta3], degrees=False).as_matrix()
    return R


def rotation_matrix_2(theta):
    """
    Returns a rotation matrix that rotates a vector by theta around the z-axis.
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R


def align_evecs(evecs, H=None, axis1=[1, 0, 0], axis2=[0, 1, 0], axis3=[0, 0, 1]):
    """
    align degenerate eigenvectors to the principal axes of the tensor.
    params:
        evecs: degenerate eigenvectors. each coloumn is an eigenvector.
        H: Hamiltonian matrix.
        axis1, axis2, axis3: eigen vectors will be  aligned
    """
    v_axis = np.array([axis1, axis2, axis3], dtype=float).T
    natom3, ndeg = evecs.shape
    natom = natom3 // 3
    v = np.tile(v_axis, (natom, 1))
    v /= np.linalg.norm(v, axis=0)[np.newaxis, :]

    # ndeg: number of degeneracy
    # nparam: number of parameters in the rotation matrix function.
    if ndeg == 3:
        rotation_matrix = rotation_matrix_3
        nparam = 3
    elif ndeg == 2:
        rotation_matrix = rotation_matrix_2
        nparam = 1
    else:
        raise ValueError("ndeg must be 2 or 3")

    # target function:
    def target(args):
        rotated_evecs = evecs @ rotation_matrix(*args)
        projections = np.einsum("ji,jk->ik", np.abs(rotated_evecs), np.abs(v))
        t = np.sum(np.abs(np.abs(projections) ** 4).real, axis=None)
        return -t

    init = np.zeros(nparam)
    bounds = [(-np.pi, np.pi)] * nparam
    res = minimize(target, init, bounds=bounds)
    args = res.x
    R = rotation_matrix(*args)
    revecs = evecs @ R
    return revecs


def detect_degeneracy(evals, tol=1e-6):
    """
    Detect degeneracy in eigenvalues.
    """
    degeneracy = []
    for i, ev in enumerate(evals):
        if i == 0:
            continue
        if np.abs(ev - evals[i - 1]) < tol:
            degeneracy.append(i)
    return degeneracy


def align_all_degenerate_eigenvectors(evals, evecs, H=None, tol=1e-4):
    aligned_evecs = copy.deepcopy(evecs)
    degenerate_with_previous = False
    ind_degenerate = []
    for i, ev in enumerate(evals):
        if i == 0:
            continue
        if np.abs(ev - evals[i - 1]) < tol:
            degenerate_with_previous = True
            if i - 1 not in ind_degenerate:
                ind_degenerate.append(i - 1)
            ind_degenerate.append(i)
        else:
            if not degenerate_with_previous:
                # previous is not degenerate
                pass
            else:
                evecs_degenerate = evecs[:, ind_degenerate]
                # print("aligning degenerate eigenvectors: ", ind_degenerate)
                # print("evecs_degenerate: ", evecs_degenerate)
                aligned_evecs[:, ind_degenerate] = align_evecs(evecs_degenerate, H=H)
                # print("aligned_evecs: ", aligned_evecs[:, ind_degenerate])
            degenerate_with_previous = False
            ind_degenerate = []

    return aligned_evecs


def test_align_evecs():
    d = np.load("Hk_short.npy")
    # H=d[21] #Gamma
    H = d[0]
    evals, evecs = eigh(H)
    print(evals)
    aligned_evecs = align_all_degenerate_eigenvectors(evals, evecs, H=H)
    tevecs = evecs[:, 1:3]
    tevals = evals[1:3]
    print(tevals)
    revecs = align_evecs(tevecs, H)
    print(revecs)


if __name__ == "__main__":
    # pass
    test_align_evecs()

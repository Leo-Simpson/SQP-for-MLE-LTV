import casadi as ca 
import numpy as np 
import numpy.linalg as LA 
from scipy.linalg.lapack import dpotrf, dpotri # type: ignore  #### apparently this is not known by Pylance, but they mention it there: # https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi
from typing import Any, cast

# Wrapper to avoid linter warnings on casadi type implementation problems
def sym(name: str, *shape: int | tuple[int, ...]) -> ca.SX:
    sx_type = cast(Any, ca.SX)
    if len(shape) == 1 and isinstance(shape[0], tuple):
        return cast(ca.SX, sx_type.sym(name, *shape[0]))
    return cast(ca.SX, sx_type.sym(name, *shape))

def dm_eye(size: int) -> ca.DM:
    dm_type = cast(Any, ca.DM)
    return cast(ca.DM, dm_type.eye(size))

def dm2np(dm, shape="vector") -> np.ndarray:
    dm_ = cast(ca.DM, dm)
    arr = dm_.full()
    if shape == "vector":
        return arr.reshape(-1)
    elif shape == "matrix":
        return arr
    else:
        raise ValueError(f"Unknown shape {shape}, should be 'vector' or 'matrix' ")


def symmetrize(x):
    return 0.5 * (x + np.swapaxes(x, 0, 1))

def psd_inverse(m, inds, det=True):
    if m.shape[0] == 1:
        if det:
            return 1./m, np.log(m[0,0])
        else:
            return 1/m
    # see https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi for why using these function.
    cholesky, info = dpotrf(m)
    if info != 0:
        print(m)
        raise ValueError("dpotrf failed")
    inv, info = dpotri(cholesky)

    if info != 0:
        raise ValueError("dpotri failed")
    tri2sym(inv, inds)
    if det:
        logdet = 2 * np.sum(np.log(np.diag(cholesky)))
        return inv, logdet
    else:
        return inv
    # # test zone
    # inv_, logdet_ = psd_inverse_slow(m, inds)
    # print(logdet_-logdet, l1(inv-inv_))


def vecToTriu(Pvec, nx):
    if isinstance(Pvec, np.ndarray):
        return ca.DM(ca.Sparsity.upper(nx), Pvec).full()
    return ca.SX(ca.Sparsity.upper(nx), Pvec)

def triuToVec(P):
    return P[P.sparsity().makeDense()[0].get_upper()]

def vec2sym(vec, n):
    if len(vec.shape) == 2 and vec.shape[1] == n:
        return vec
    P = ca.SX.zeros(n, n)
    k = 0
    for i in range(n):
        P[i, i] = vec[k]
        k += 1
        for j in range(i+1, n):
            P[i, j] = vec[k]
            P[j, i] = vec[k]
            k += 1
    assert k == int(n*(n + 1)/ 2)
    return P

def sym2vec(P):
    n = P.shape[0]
    assert P.shape[0] == P.shape[1], "P should be square"
    vec = []
    for i in range(n):
        vec.append(P[i, i])
        for j in range(i+1, n):
            vec.append( P[i, j] )
    assert len(vec) == int(n*(n + 1)/ 2)
    return ca.vcat(vec)

def mult(L):
    output = L[0]
    for l in L[1:]:
        output = output @ l
    return output


def is_psd(x, tol=0.):
    mineig = np.min(LA.eigvals(x))
    return np.allclose(x, x.T) and mineig >= -tol
    # return np.allclose(x, x.T)
    # return mineig

def is_s(x):
    return np.allclose(x, x.T)


def psd_inverse_(A, inds, det=True):
    if det:
        return LA.inv(A), np.log(LA.det(A))
    else:
        return LA.inv(A)

def tri2sym(m, inds):
    m[inds] = m.T[inds]

def l1(x):
    return abs(x).sum()

from packaging.version import Version

def select_jac(A, nx):
    """
        Select the Jacobian block with respect to the first variable.

        CasADi <= 3.5 returns a matrix-like Jacobian, while newer versions
        return a tuple/list-like object.
    """
    if Version(ca.__version__) < Version("3.6"):
        return A[:, :nx]
    else:
        return A[0]
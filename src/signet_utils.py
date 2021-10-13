''' modified from https://github.com/alan-turing-institute/SigNet
'''
import math

import numpy as np
from scipy import optimize as opt
import scipy.sparse as ss
import sklearn.cluster as sl

def objscore(labels, k, mat1, mat2=None):
    """Scores a clustering using the objective matrix given
    Args:
        labels (list of int): Clustering assignment.
        k (int): Number of clusters.
        mat1 (csc matrix): Numerator matrix of objective score.
        mat2 (csc matrix): Denominator matrix of objective score. Default is no denominator.
    Returns:
        float: Score.
    """
    
    tot = 0
    row=np.empty(k,dtype=object)
    for pos, item in enumerate(labels):
        if type(row[item])!=list:
            row[item] = [pos]
        else:
            row[item].append(pos)
    for j in range(k):
        num = mat1[:,row[j]].tocsr()[row[j],:].sum()
        if mat2!=None:
            den= mat2[:,row[j]].tocsr()[row[j],:].sum()
            if den==0:
                den=1
            num = num/den
        tot += num
    return float(round(tot,2))

def sizeorder(labels,k,pos,neg,largest=True):
    n = len(labels)
    eye=ss.eye(n,format='csc')
    clusscores=np.empty(k)
    lsize=0
    lclus=-1
    for j in range(k):
        row = [i for i in range(n) if labels[i] == j]
        col = [0 for i in range(n) if labels[i] == j]
        dat = [1 for i in range(n) if labels[i] == j]
        if largest==False and len(dat)>lsize:
            lsize=len(dat)
            lclus=j
        vec = ss.coo_matrix((dat, (row, col)), shape=(n, 1))
        vec = vec.tocsc()
        x = vec.transpose() * pos * vec
        y = vec.transpose() * (neg+eye)* vec
        z=float(x[0,0])/float(y[0,0])
        clusscores[j] = z
    new=[x for x in range(n) if labels[x]!=lclus]
    scores = [clusscores[labels[i]] for i in new]
    return [x for _,x in sorted(zip(scores,new))]
    
def invdiag(M):
    """Inverts a positive diagonal matrix.
    Args:
        M (csc matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted diagonal
    """

    d = M.diagonal()
    dd = [1 / max(x, 1 / 999999999) for x in d]
    return ss.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


def sqrtinvdiag(M):
    """Inverts and square-roots a positive diagonal matrix.
    Args:
        M (csc matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return ss.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


def merge(elemlist):
    """Merges pairs of clusters randomly. 
    Args:
        elemlist (list of lists of int): Specifies the members of each cluster in the current clustering
    Returns:
        list of lists of int: New cluster constituents
        boolean: Whether last cluster was unable to merge
        list of int: List of markers for current clustering, to use as starting vectors.
    """
    k = len(elemlist)
    dc = False
    elemlist.append([])
    perm = np.random.permutation(k)
    match = [k] * k
    for i in range(math.floor(k / 2)):
        me = perm[2 * i]
        you = perm[2 * i + 1]
        match[me] = you
        match[you] = me
    if k % 2 != 0:
        dontcut = perm[k - 1]
        dc = True
    nelemlist = [elemlist[i] + elemlist[match[i]] for i in range(k) if i < match[i] < k]
    numbers = [len(elemlist[i]) for i in range(k) if i < match[i] < k]
    if dc:
        nelemlist.append(elemlist[dontcut])
    return nelemlist, dc, numbers

def cut(elemlist, matrix, numbers, dc,mini):
    """Cuts clusters by separately normalised PCA.
    Args:
        elemlist (list of lists of int): Specifies the members of each cluster in the current clustering
        matrix (csc matrix): Matrix objective with which to cut.
        numbers (list of int): Marks previous clustering to use as starting vector.
        dc (boolean): Whether to skip cutting last cluster
        mini (boolean): Whether to minimise (instead of maximise) matrix objective.
    Returns:
        list of lists of int: new cluster constituents
    """
    nelemlist = []
    if dc:
        nelemlist.append(elemlist.pop())
    count = 0
    for i in elemlist:
        l = len(i)
        if l > 2:
            matrix1 = matrix[:, i].tocsr()
            matrix1 = matrix1[i, :].tocsc()
            val = 1 / math.sqrt(l)
            v = [-val] * numbers[count]
            w = [val] * (l - numbers[count])
            v = v + w
            if not mini:
                (w, v) = ss.linalg.eigsh(matrix1, 2, which='LA', maxiter=l, v0=v)
            else:
                (w, v) = ss.linalg.eigsh(matrix1, 2, which='SA', maxiter=l, v0=v)
            x = sl.KMeans(n_clusters=2,n_init=3,max_iter=100).fit(v)
            c1 = [i[y] for y in range(l) if x.labels_[y]==0]
            c2 = [i[y] for y in range(l) if x.labels_[y]==1]
            nelemlist.append(c1)
            nelemlist.append(c2)
        elif len(i) == 2:
            if matrix[i[0], i[1]] > 0:
                nelemlist.append(i)
                nelemlist.append([])
            else:
                nelemlist.append([i[0]])
                nelemlist.append([i[1]])
        elif len(i) == 1:
            nelemlist.append(i)
            nelemlist.append([])
        else:
            nelemlist.append([])
            nelemlist.append([])
        count += 1
    return nelemlist


def augmented_lagrangian(A, r, printing=False, init=None):
    """Augmented Lagrangian optimisation of the BM problem.
    It finds the matrix X which maximises the Frobenius norm (A, X.dot(X.T))
    with the constraint of having unit elements along the diagonal of X.dot(X.T).
    Args:
        A (csc matrix): The adjacency matrix
        r (int): The rank of the final solution
        printing (bool): Whether to print optimisation information
        init (array): Initial guess for the solution. If None a random matrix is used.
    Returns:
        array: The optimal matrix of dimensions n x r
    """

    n, _ = A.shape
    y = np.ones(n).reshape((-1, 1))
    if init is None:
        X = np.random.uniform(-1, 1, size=(n, r))
    else:
        X = init
    penalty = 1
    gamma = 10
    eta = .25
    target = .01  # 0.01
    vec = _constraint_term_vec(n, X)
    v = vec.reshape((1, -1)).dot(vec)
    v_best = v
    while v > target:
        Rv = _matrix_to_vector(X)
        if printing == True:
            print('Starting L-BFGS-B on augmented Lagrangian..., v is ', v)
        optimizer = opt.minimize(lambda R_vec: _augmented_lagrangian_func(
            R_vec, A, y, penalty, n, r), Rv, jac=lambda R_vec: _jacobian(R_vec, A, n, y, penalty, r), method="L-BFGS-B")
        if printing == True:
            print('Finishing L-BFGS-B on augmented Lagrangian...')
        X = _vector_to_matrix(optimizer.x, r)
        vec = _constraint_term_vec(n, X)
        v = vec.reshape((1, -1)).dot(vec)
        if printing == True:
            print('Finish updating variables...')
        if v < eta * v_best:
            y = y - penalty * vec
            v_best = v
        else:
            penalty = gamma * penalty
    if printing == True:
        print('Augmented Lagrangian terminated.')
    return X


def _generate_random_rect(n, k):
    """
    Returns a random initialization of matrix.
    """

    X = np.random.uniform(-1, 1, (n, k))
    for i in range(n):
        X[i, :] = X[i, :] / np.linalg.norm(X[i, :])
    return X


def _basis_vector(size, index):
    """
    Returns a basis vector with 1 on certain index.
    """

    vec = np.zeros(size)
    vec[index] = 1
    return vec


def _trace_vec(X):
    """
    Returns a vector containing norm square of row vectors of X.
    """

    vec = np.einsum('ij, ij -> i', X, X)

    return vec.reshape((-1, 1))


def _constraint_term_vec(n, X):
    """
    Returns the vector required to compute objective function value.
    """

    vec = _trace_vec(X)
    constraint = vec - np.ones(n).reshape((-1, 1))

    return constraint


def _augmented_lagrangian_func(Xv, A, y, penalty, n, k):
    """
    Returns the value of objective function of augmented Lagrangian.
    """

    X = _vector_to_matrix(Xv, k)

    vec = _constraint_term_vec(n, X)

    AX = A.dot(X)

    objective1 = - np.einsum('ij, ij -> ', X, AX)  # Trace(Y*X*X.T)

    objective2 = - y.reshape((1, -1)).dot(vec)

    objective3 = + penalty / 2 * vec.reshape((1, -1)).dot(vec)

    objective = objective1 + objective2 + objective3

    return objective


def _vector_to_matrix(Xv, k):
    """
    Returns a matrix from reforming a vector.
    """
    U = Xv.reshape((-1, k))
    return U


def _matrix_to_vector(X):
    """
    Returns a vector from flattening a matrix.
    """

    u = X.reshape((1, -1)).ravel()
    return u


def _jacobian(Xv, Y, n, y, penalty, k):
    """
    Returns the Jacobian matrix of the augmented Lagrangian problem.
    """

    X = _vector_to_matrix(Xv, k)

    vec_trace_A_ = _trace_vec(X).ravel() - 1.

    vec_second_part = np.einsum('ij, i -> ij', X, y.ravel())

    vec_third_part = np.einsum('ij, i -> ij', X, vec_trace_A_)

    jacobian = - 2 * Y.dot(X) - 2 * vec_second_part + \
               2 * penalty * vec_third_part

    jac_vec = _matrix_to_vector(jacobian)
    return jac_vec.reshape((1, -1)).ravel()
import numpy as np

"""
Implementation of utility and symmetry functions in Michele Parrinello's paper

https://aiichironakano.github.io/cs653/Behler-NNPES-PRL07.pdf
"""
def pairwise_vec(coords:np.ndarray):
    return coords - coords[:, np.newaxis]

def pairwise_dist(coords:np.ndarray):
    R = coords - coords[:, np.newaxis]
    return np.squeeze(np.sqrt(np.sum(R**2, axis=-1, keepdims=True)))


def pairwise_dist_area_element(coords:np.ndarray): # 30xN_edgex2
    n_area_element, n_edge, _ = coords.shape

    for edge in range(n_edge-2):

        r1 = np.array(coords[:, edge, :] - coords[:, edge+1, :]).reshape(n_area_element, 1, -1)
        r2 = np.array(coords[:,edge+2,:] - coords[:, edge+1, :]).reshape(n_area_element, 1, -1)
        r1_norm = np.linalg.norm(r1, axis=-1)
        r2_norm = np.linalg.norm(r2, axis=-1)
        angle = np.arccos(np.einsum('ijk,ijk->ij', r1, r2).squeeze() / np.einsum('ij,ij->i', r1_norm, r2_norm)).reshape(n_area_element, -1)

        if edge == 0:
            descriptor = np.concatenate((r1_norm, angle), axis=1)
        else:
            descriptor = np.concatenate((descriptor.reshape(n_area_element, -1), r1_norm, angle),axis=1)

    r1 = (coords[:, n_edge-2,:] -coords[:, n_edge-1,:]).reshape(n_area_element, 1, -1)
    r1_norm = np.linalg.norm(r1, axis=-1)
    r2 = (coords[:, 0, :] - coords[:, n_edge-1, :]).reshape(n_area_element, 1, -1)
    r2_norm = np.linalg.norm(r2, axis=-1)

    angle = np.arccos(np.einsum('ijk,ijk->ij', r1, r2).squeeze() / np.einsum('ij,ij->i', r1_norm, r2_norm)).reshape(n_area_element, -1)
    descriptor = np.concatenate((descriptor.reshape(n_area_element,-1), r1_norm, angle), axis=1)

    r1 = (coords[:, n_edge-1, :] - coords[:, 0, :]).reshape(n_area_element, 1, -1)
    r1_norm = np.linalg.norm(r1, axis=-1)
    r2 = (coords[:,1,:] - coords[:, 0, :]).reshape(n_area_element, 1, -1)
    r2_norm = np.linalg.norm(r2, axis=-1)

    angle =np.arccos(np.einsum('ijk,ijk->ij', r1, r2).squeeze() / np.einsum('ij,ij->i', r1_norm, r2_norm)).reshape(n_area_element, -1)
    descriptor = np.concatenate((descriptor.reshape(n_area_element, -1), r1_norm, angle), axis=1)

    return descriptor


def cutoff(R:np.ndarray, R_c=10):
    return np.where(R <= R_c, 0.5 * (np.cos(np.pi * R / R_c) + 1), 0)


def radial_sym(R:np.ndarray, R_s=0., eta=0.01):
    gaussian = np.exp(-eta * (R - R_s)**2)
    f_c = cutoff(R)
    G = gaussian * f_c
    return np.sum(G - np.diag(G.diagonal()), axis=0)
    # return np.sum(G , axis=0)


def angular_sym(vec:np.ndarray, R:np.ndarray, lamb=1, eta=0.01, ksi=1):
    G = []
    for i, v in enumerate(vec): 
        # Here i is literally index i, v (2D array) contains the j-k elements
        # In below calculation, we can treat i-th atom fixed
        v_dot_v = np.dot(v, v.T).astype(float)
        r_times_r = np.reshape([R[i][j]*R[i][k] for j in range(len(R[i])) for k in range(len(R[i]))], R.shape)
        # print("vector:\n", vec)
        # print("pairwise dot product:\n", v_dot_v)
        # print("pairwise separation sq:\n", r_times_r)
        # For cos_theta:
        # Note1: i fixed, stores j-k element cos(theta) of R_ij and R_ik, 
        # Note2: i-th row and col outputing nan is normal, as R_ii is a zero vector i.e. a point (R_ii dot R_ij = R_ii dot R_ik = nan)
        # Note3: diagonal = 1 is normal, as self dot product is 0 degrees ~ cos(0) = 1 (R_ij dot R_ik where j = k)

        cos_theta = np.divide(v_dot_v, r_times_r, out=np.full(v_dot_v.shape, np.nan), where=(r_times_r != 0))
        # cos_theta = np.cos(np.deg2rad(theta))
        angular = (1 + lamb*cos_theta)**ksi
        # print("cos_theta:\n", cos_theta)
        # print("angular term:\n", angular)
        # Cutoff array, accessible with i, j, k
        cut = cutoff(R)
        # print("cutoff:\n", cut)
        for j in range(len(R[i])):
            for k in range(len(R[i])):
                if (j == i) or (k == i): continue
                # Element-wise multiplication of angular term with gaussian and cutoff functions
                # I have to use nested loop because the element access from different arrays is difficult to vectorize
                angular[j][k] *= np.exp(-eta * (R[i][j]**2 + R[i][k]**2 + R[j][k]**2)) * cut[i][j] * cut[i][k] * cut[j][k]
        # print("angular times gaussian term:\n", angular)
        G.append(2**(1-ksi) * np.nansum(angular))
    return np.array(G)

def my_ang_sym_func(theta, lamb=1, eta=3, zeta=1):

    cos_theta = np.cos(np.deg2rad(theta))
    angular = (1 + lamb * cos_theta) ** zeta
    cut = 0.5*(np.cos(np.pi/10)+1)
    angular_jk = angular * np.exp(-eta * (3 ** 2 + 3 ** 2 + 3 ** 2)) * cut * cut * cut
    G = 2 ** (1 - zeta) * angular_jk * 2

    return G

if __name__ == "__main__":

    R_s = [2., 3., 4., 5., 6., 7., 8., 9., 10.]
    eta = [0.01, 0.03, 0.06, 0.10, 0.20, 0.40, 1.00, 5.00]


    R = np.array([0.05*i for i in range(300)])

    # # coords = np.arange(10).reshape((-1, 2))
    # np.random.seed(11)
    # coords = np.random.randint(10, size=(5, 2))
    # R = pairwise_dist(coords)
    # vec = pairwise_vec(coords)
    # print("coords\n", coords)
    # print("pairwise dist\n", R)
    # # print(vec)
    # print("radial sym func\n", radial_sym(R))
    # print("angular sym func\n", angular_sym(vec, R))

    import matplotlib.pyplot as plt

    fig = plt.figure()

    # plt.xlabel("Rij")
    # plt.ylabel("Radia sym func, Rs=0")
    #
    # for each_eta in eta:
    #     G_rad = [radial_sym(r, R_s=0, eta=each_eta) for r in R]
    #     plt.plot(R, G_rad, label=f"eta={each_eta}")
    #
    # plt.legend()
    #
    # plt.show()

    # plt.xlabel("Rij")
    # plt.ylabel("Radia sym func, Rs>0")
    #
    # for rs in R_s:
    #     G_rad = [radial_sym(r, R_s=rs, eta=1) for r in R]
    #     plt.plot(R, G_rad, label=f"Rs={rs}")
    #
    # plt.legend()
    # plt.show()

    zeta = [1, 2, 4, 16, 64]
    theta = list(range(0, 360, 1))

    plt.xlabel("theta (degree)")
    plt.ylabel("Anglar sym function, lamba=-1")
    # plt.title("Angular sym func for a triatomic system")

    for i in zeta:
        G = [my_ang_sym_func(t,lamb=-1,eta=5, zeta=i) for t in theta]
        plt.plot(theta, G, label=f"zeta={i}")

    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import multivariate_normal as mvn
import pdb

#|%%--%%| <IwmgquwXVi|3tT4JeanSm>
r"""°°°
Homework 5
°°°"""
#|%%--%%| <3tT4JeanSm|uRDY4X4WDS>
r"""°°°
Problem 1
°°°"""
#|%%--%%| <uRDY4X4WDS|dN3OVRtBbt>
r"""°°°
Define the gaussian mixture model
°°°"""
#|%%--%%| <dN3OVRtBbt|jNeRg9gWL2>

w1, w2, w3 = 0.5, 0.2, 0.3
mew1, mew2, mew3  = np.array((0.35, 0.38)), np.array((0.68, 0.25)), np.array((0.56, 0.64))
sigma1 = np.array(((0.01, 0.004),(0.004, 0.01)))
sigma2 = np.array(((0.005, -0.003), (-0.003, 0.005)))
sigma3 = np.array(((0.008, 0.0), (0.0, 0.004)))

def pdf(x):
    return w1 * mvn.pdf(x, mew1, sigma1) + \
           w2 * mvn.pdf(x, mew2, sigma2) + \
           w3 * mvn.pdf(x, mew3, sigma3)

#|%%--%%| <jNeRg9gWL2|CLF337ZSew>
r"""°°°
Define Ergodic parameters
°°°"""
#|%%--%%| <CLF337ZSew|HUfOhMaWIM>

### We are going to use 10 coefficients per dimension --- so 100 index vectors in total

num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T  # this is the set of all index vectors
print('First 5 index vectors: ')
print(ks[:5, :])

# define a 1-by-1 2D search space
L_list = np.array([1.0, 1.0])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 100),
    np.linspace(0, L_list[1], 100)
)

grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
pdf_gt = pdf(grids)  # ground truth density function
dx = 1.0 / 99
dy = 1.0 / 99  # the resolution of the grids

#|%%--%%| <HUfOhMaWIM|7oTcm75DTY>
r"""°°°
Define iLQR Parameters
°°°"""
#|%%--%%| <7oTcm75DTY|FBcyeniF8x>

q = 0.03
dt = 0.1
tsteps = 100
T = tsteps * dt
init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps,1))

Q_x = np.diag([0.1, 0.1])
R_u = np.diag([0.001, 0.001])
P1 = np.diag([20.0, 20.0])

# Q_z = np.diag([0.1, 0.1])
# R_v = np.diag([0.1, 0.1])
Q_z = Q_x
R_v = R_u

#|%%--%%| <FBcyeniF8x|ykzjSeDrA9>
r"""°°°
Define iLQR, Dynamics, and other helper functions
°°°"""
#|%%--%%| <ykzjSeDrA9|GBajMC2FFA>

def dyn(xt, ut):
    xdot = ut
    return xdot

def step(xt, ut):
    k1 = dt * dyn(xt, ut)
    k2 = dt * dyn(xt + 0.5 * k1, ut)
    k3 = dt * dyn(xt + 0.5 * k2, ut)
    k4 = dt * dyn(xt + k3, ut)
    xt_new = xt + (1 / 6) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return xt_new

def loss(ergodic_metric, u_traj):
    ut_sum = 0
    for ut in u_traj:
        ut_sum += ut.T @ R_u @ ut * dt
    return q * ergodic_metric + 1/2 * ut_sum

def traj_sim(x0, ulist):
    tsteps = ulist.shape[0]
    x_traj = np.zeros((tsteps, 2)) # should this stay with 3 columns?
    xt = x0.copy()
    for t in range(tsteps):
        xt_new = step(xt, ulist[t])
        x_traj[t] = xt_new.copy()
        xt = xt_new.copy()
    return x_traj

# correct
def get_A(t, xt, ut):
    A = np.zeros((2,2))
    return A

#correct
def get_B(t, xt, ut):
    B = np.eye(2)

    return B

# really unsure about this
def dldx(t, ck, phik, lambda_k, st):
    dvec = 0
    Dfk_vals = []
    for i, k_vec in enumerate(ks):
        Dfk = np.pi * k_vec / L_list * -np.sin(np.pi * k_vec / L_list * st)
        hk = np.sqrt(np.sum(np.square(np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1))) * dx * dy)
        Dfk /= hk
        Dfk_vals.append(Dfk)

    Dfk_vals = np.array(Dfk_vals)
    for i in range(ks.shape[0]):
        dvec += lambda_k[i] * (2/T * ck[i] - phik[i]) * Dfk_vals[i]
    dvec *= q
    return dvec


def dldu(t, xt, ut):
    dvec = ut @ R_u
    return dvec
    
#|%%--%%| <GBajMC2FFA|Mi3cMCfCyX>
r"""°°°
Function for iLQR Iterations
°°°"""
#|%%--%%| <Mi3cMCfCyX|Dpd710YJVF>

def ilqr_iter(x0, u_traj, traj_coefficients, dist_coefficients, lambda_k, s_traj):
    """
    :param x0: initial state of the system
    :param u_traj: current estimation of the optimal control trajectory
    :return: the descent direction for the control
    """
    # forward simulate the state trajectory
    x_traj = traj_sim(x0, u_traj)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((tsteps, 2, 2))
    B_list = np.zeros((tsteps, 2, 2))
    a_list = np.zeros((tsteps, 2))
    b_list = np.zeros((tsteps, 2))
    for t_idx in range(tsteps):
        t = t_idx * dt
        A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
        B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, traj_coefficients, dist_coefficients, lambda_k, s_traj[t_idx])
        b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])


    xd_T = np.array([
        2.0*(tsteps-1)*dt / np.pi, 0.0, np.pi/2.0
    ])  # desired terminal state
    p1 = 2 * P1 @ (x_traj[-1] - xd_T[-1]) # probably wrong!!

    def zp_dyn(t, zp):
        t_idx = (t/dt).astype(int)
        At = A_list[t_idx]
        Bt = B_list[t_idx]
        at = a_list[t_idx]
        bt = b_list[t_idx]

        M_11 = At
        M_12 = -Bt @ np.linalg.inv(R_v) @ Bt.T
        M_21 = -Q_z
        M_22 = -At.T
        dyn_mat = np.block([
            [M_11, M_12],
            [M_21, M_22]
        ])

        m_1 = -Bt @ np.linalg.inv(R_v) @ bt
        m_2 = -at
        dyn_vec = np.hstack([m_1, m_2])

        return dyn_mat @ zp + dyn_vec

    # this will be the actual dynamics function you provide to solve_bvp,
    # it takes in a list of time steps and corresponding [z(t), p(t)]
    # and returns a list of [zdot(t), pdot(t)]
    def zp_dyn_list(t_list, zp_list):
        list_len = len(t_list)
        zp_dot_list = np.zeros((4, list_len))
        for _i in range(list_len):
            zp_dot_list[:,_i] = zp_dyn(t_list[_i], zp_list[:,_i])
        return zp_dot_list

    # boundary condition (inputs are [z(0),p(0)] and [z(T),p(T)])
    def zp_bc(zp_0, zp_T):
        return np.array([zp_0[:2], zp_T[:2]]).flatten()

    ### The solver will say it does not converge, but the returned result
    ### is numerically accurate enough for our use
    # zp_traj = np.zeros((tsteps,6))  # replace this by using solve_bvp
    tlist = np.arange(tsteps) * dt
    res = solve_bvp(
        zp_dyn_list, zp_bc, tlist, np.zeros((4,tsteps)),
        max_nodes=100
    )
    zp_traj = res.sol(tlist).T

    z_traj = zp_traj[:,:2]
    p_traj = zp_traj[:,2:]

    v_traj = np.zeros((tsteps, 2))
    for _i in range(tsteps):
        At = A_list[_i]
        Bt = B_list[_i]
        at = a_list[_i]
        bt = b_list[_i]

        zt = z_traj[_i]
        pt = p_traj[_i]

        vt = -np.linalg.inv(R_v) @ (Bt.T @ pt + bt)
        v_traj[_i] = vt

    return v_traj

#|%%--%%| <Dpd710YJVF|0wOwj2lSOF>
r"""°°°
Begin iLQR Iterations
°°°"""
#|%%--%%| <0wOwj2lSOF|uea1Kt4g4G>

init_u_traj = np.tile(np.array([0.05, 0.02]), reps=(tsteps,1))

x0 = np.ones(2) * 0.3
u_traj = init_u_traj.copy()
x_traj_initial = None
loss_list = []
time = np.arange(tsteps) * dt
fig, ax = plt.subplots(3,1, figsize=(6, 10))
plt.subplots_adjust(hspace=0.5)

# Compute the coefficients of the spatial distribution, \phi_k
# I think these will be the same for all iterations
dist_coefficients = np.zeros(ks.shape[0])  # number of coefficients matches the number of index vectors
for i, k_vec in enumerate(ks):
    # step 1: evaluate the fourier basis function over all the grid cells
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  # we use NumPy's broadcasting feature to simplify computation
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)  # normalization term
    fk_vals /= hk

    # step 2: evaluate the spatial probabilty density function over all the grid cells
    pdf_vals = pdf(grids)  # this can computed ahead of the time

    # step 3: approximate the integral through the Riemann sum for the coefficient
    phik = np.sum(fk_vals * pdf_vals) * dx * dy
    dist_coefficients[i] = phik

# Start iLQR iterations here
for iter in range(10):
    s_traj = traj_sim(x0, u_traj)
    if iter == 0:
        s_traj_initial = s_traj.copy()
    
    # compute the coefficient of the trajectory
    traj_coefficients = np.zeros(ks.shape[0])
    for i, k_vec in enumerate(ks):
        # step 1: evaluate the basis function over the trajectory
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * s_traj), axis=1)
        hk = np.sqrt(np.sum(np.square(np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1))) * dx * dy)
        fk_vals /= hk

        # step 2: approximate the integral through the Riemann sum for the coefficient
        ck = np.sum(fk_vals) * dt / (tsteps * dt)
        traj_coefficients[i] = ck

    # Finally, we compute the erogdic metric
    lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
    erg_metric = np.sum(lamk_list * np.square(traj_coefficients - dist_coefficients))

    # iLQR stuff now

    # get descent direction
    v_traj = ilqr_iter(x0, u_traj, traj_coefficients, dist_coefficients, lamk_list, s_traj)

    # Armijo line search parameters
    gamma = 1.0  # initial step size
    alpha = 1e-04
    beta = 0.5

    total_loss = loss(erg_metric, u_traj)
    total_other_loss = loss(erg_metric, u_traj + gamma * v_traj)
    loss_list.append(total_loss)

    # while total_other_loss > total_loss + alpha * gamma * np.sum(v_traj.T @ v_traj):
    #     gamma = beta * gamma
    #     total_other_loss = loss(erg_metric, u_traj + gamma * v_traj)
    #     print(gamma)
    gamma = 1.0

    # update control for the next iteration
    u_traj += gamma * v_traj

    # plt.plot(s_traj_initial[:,0], s_traj_initial[:,1], linestyle='--', color='k', label="Initial Trajectory")
    # plt.plot(s_traj[:,0], s_traj[:,1], linestyle='-', color='k', label="Converged Trajectory")
    # plt.legend(loc='upper right')
    # plt.xlim(0.0, L_list[0])
    # plt.ylim(0.0, L_list[1])
    # plt.title('Original PDF')
    # plt.contourf(grids_x, grids_y, pdf_gt.reshape(grids_x.shape), cmap='Reds')
    # plt.show()

ax[0].plot(s_traj_initial[:,0], s_traj_initial[:,1], linestyle='--', color='k', label="Initial Trajectory")
ax[0].plot(s_traj[:,0], s_traj[:,1], linestyle='-', color='k', label="Converged Trajectory")
ax[0].set_title('State Trajectory')
ax[0].legend(loc='upper right')
ax[0].set_xlim(0.0, L_list[0])
ax[0].set_ylim(0.0, L_list[1])
ax[0].set_title('Original PDF')
ax[0].contourf(grids_x, grids_y, pdf_gt.reshape(grids_x.shape), cmap='Reds')


ax[1].plot(time, u_traj[:,0], label='u1')
ax[1].plot(time, u_traj[:,1], label='u2')
ax[1].legend()
ax[1].set_xlim(0, 6)
ax[1].set_ylim(-3, 3)
ax[1].set_title('Optimal Control')

ax[2].plot(np.arange(10), loss_list, label='loss')
ax[2].set_title('Objective Value')
ax[2].set_xlabel('Iteration')
ax[2].set_ylabel('Objective')
ax[2].set_xlim(0, 9)
ax[2].set_ylim(0, 100)

plt.savefig('hw5_1.png')
plt.show()

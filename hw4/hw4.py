import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import pdb

#|%%--%%| <9s2IUoUm8h|FkYdSakqDn>
r"""°°°
Problem 1
°°°"""
#|%%--%%| <FkYdSakqDn|QGiJKkgPO2>



#|%%--%%| <QGiJKkgPO2|u9c6RLyAu6>
r"""°°°
Problem 2
°°°"""
#|%%--%%| <u9c6RLyAu6|ULtqp6mH6I>

def armijo(xk, gamma0, alpha, beta):
    gamma = gamma0
    grad_J = grad_f(xk)
    zk = -grad_J
    while f(xk + gamma * zk) > f(xk) - alpha * gamma * grad_J @ zk:
        gamma = beta * gamma
    
    return xk + gamma * zk

def f(xk):
    x1 = xk[0]
    x2 = xk[1]
    return 0.26 * (x1**2 + x2**2) - 0.46 * x1 * x2

def grad_f(xk):
    x1 = xk[0] 
    x2 = xk[1]
    return np.array([0.52 * x1 - 0.46 * x2, -0.46 * x1 + 0.52 * x2])


#|%%--%%| <ULtqp6mH6I|NJgmDxKlFd>

num_iter = 100 

xk = np.array([-4, -2])
xk_list = [xk]
xk_initial = xk
gamma0 = 1
alpha = 10**-4
beta = 0.5
for i in range(num_iter):
    xk = armijo(xk, gamma0, alpha, beta)
    xk_list.append(xk)

xk_final = xk_list[-1]

x_grid = np.linspace(-5, 5, 100)
y_grid = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_grid, y_grid)
gradients = np.array([grad_f(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())])
U = gradients[:,0].reshape(X.shape)
V = gradients[:,1].reshape(X.shape)

# plot xk_initial, xk_final, and the path xk_list
xk_list = np.array(xk_list)
plt.plot(xk_list[:,0], xk_list[:,1], '-')
plt.plot(xk_initial[0], xk_initial[1], 'ro')
plt.plot(xk_final[0], xk_final[1], 'go')
plt.contourf(X, Y, f(np.array([X,Y])), 100, alpha=0.5)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Armijo Line Search')
plt.show()

# plt.savefig('armijo_line_search.png')

#|%%--%%| <NJgmDxKlFd|IwmgquwXVi>
r"""°°°
Problem 3
°°°"""
#|%%--%%| <IwmgquwXVi|pxXurdf3bT>

# define parameters
dt = 0.1
x0 = np.array([0.0, 0.0, np.pi/2.0])
tsteps = 63
init_u_traj = np.tile(np.array([1.0, -0.5]), reps=(tsteps,1))

Q_x = np.diag([10.0, 10.0, 2.0])
R_u = np.diag([4.0, 2.0])
P1 = np.diag([20.0, 20.0, 5.0])

Q_z = np.diag([5.0, 5.0, 1.0])
R_v = np.diag([2.0, 1.0])

#|%%--%%| <pxXurdf3bT|PHHpmH2yCb>

def dyn(xt, ut):
    xdot = np.array([np.cos(xt[2]) * ut[0],
                     np.sin(xt[2]) * ut[0],
                     ut[1]])
    return xdot


def get_A(t, xt, ut):
    A = np.array([[0.0, 0.0, -np.sin(xt[2]) * ut[0]],
                  [0.0, 0.0, np.cos(xt[2]) * ut[0]],
                  [0.0, 0.0, 0.0]])
    return A


def get_B(t, xt, ut):
    B = np.array([[np.cos(xt[2]), 0.0],
                  [np.sin(xt[2]), 0.0],
                  [0.0, 1.0]])
    return B

def step(xt, ut):
    xt_new = xt + dt * dyn(xt, ut)  # recommended: replace it with RK4 integration
    return xt_new


def traj_sim(x0, ulist):
    tsteps = ulist.shape[0]
    x_traj = np.zeros((tsteps, 3))
    xt = x0.copy()
    for t in range(tsteps):
        xt_new = step(xt, ulist[t])
        x_traj[t] = xt_new.copy()
        xt = xt_new.copy()
    return x_traj


def loss(t, xt, ut):
    xd = np.array([
        2.0*t / np.pi, 0.0, np.pi/2.0
    ])  # desired system state at time t

    x_loss = (xt - xd).T @ Q_x @ (xt - xd)
    u_loss = ut.T @ R_u @ ut

    return x_loss + u_loss


def dldx(t, xt, ut):
    xd = np.array([
        2.0*t / np.pi, 0.0, np.pi/2.0
    ])

    dvec = 2 * Q_x @ (xt - xd)
    return dvec


def dldu(t, xt, ut):
    dvec = 2 * R_u @ ut
    return dvec

#|%%--%%| <PHHpmH2yCb|lPXAoBSw0C>

def ilqr_iter(x0, u_traj):
    """
    :param x0: initial state of the system
    :param u_traj: current estimation of the optimal control trajectory
    :return: the descent direction for the control
    """
    # forward simulate the state trajectory
    x_traj = traj_sim(x0, u_traj)

    # compute other variables needed for specifying the dynamics of z(t) and p(t)
    A_list = np.zeros((tsteps, 3, 3))
    B_list = np.zeros((tsteps, 3, 2))
    a_list = np.zeros((tsteps, 3))
    b_list = np.zeros((tsteps, 2))
    for t_idx in range(tsteps):
        t = t_idx * dt
        A_list[t_idx] = get_A(t, x_traj[t_idx], u_traj[t_idx])
        B_list[t_idx] = get_B(t, x_traj[t_idx], u_traj[t_idx])
        a_list[t_idx] = dldx(t, x_traj[t_idx], u_traj[t_idx])
        b_list[t_idx] = dldu(t, x_traj[t_idx], u_traj[t_idx])

    xd_T = np.array([
        2.0*(tsteps-1)*dt / np.pi, 0.0, np.pi/2.0
    ])  # desired terminal state
    p1 = 2 * P1 @ (x_traj[-1] - xd_T[-1])

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

        m_1 = -Bt @ np.linalg.inv(R_v) @ bt.T
        m_2 = -at
        dyn_vec = np.hstack([m_1, m_2])

        return dyn_mat @ zp + dyn_vec

    # this will be the actual dynamics function you provide to solve_bvp,
    # it takes in a list of time steps and corresponding [z(t), p(t)]
    # and returns a list of [zdot(t), pdot(t)]
    def zp_dyn_list(t_list, zp_list):
        list_len = len(t_list)
        zp_dot_list = np.zeros((6, list_len))
        for _i in range(list_len):
            zp_dot_list[:,_i] = zp_dyn(t_list[_i], zp_list[:,_i])
        return zp_dot_list

    # boundary condition (inputs are [z(0),p(0)] and [z(T),p(T)])
    def zp_bc(zp_0, zp_T):
        return np.array([zp_0[:3], zp_T[:3]]).flatten()

    ### The solver will say it does not converge, but the returned result
    ### is numerically accurate enough for our use
    # zp_traj = np.zeros((tsteps,6))  # replace this by using solve_bvp
    tlist = np.arange(tsteps) * dt
    res = solve_bvp(
        zp_dyn_list, zp_bc, tlist, np.zeros((6,tsteps)),
        max_nodes=100
    )
    zp_traj = res.sol(tlist).T

    z_traj = zp_traj[:,:3]
    p_traj = zp_traj[:,3:]

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

#|%%--%%| <lPXAoBSw0C|FBcyeniF8x>

# Start iLQR iterations here

u_traj = init_u_traj.copy()
x_traj_initial = None
loss_list = []
time = np.arange(tsteps) * dt
fig, ax = plt.subplots(3,1, figsize=(3, 5))
plt.subplots_adjust(hspace=0.5)
for iter in range(10):
    # forward simulate the current trajectory
    x_traj = traj_sim(x0, u_traj)
    if iter == 0:
        x_traj_initial = x_traj.copy()

    # visualize the current trajectory
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title('Iter: {:d}'.format(iter))
    # ax.set_aspect('equal')
    # ax.set_xlim(-0.2, 4.2)
    # ax.set_ylim(-0.2, 2.2)
    # ax.plot(x_traj[:,0], x_traj[:,1], linestyle='-', color='C0')
    # plt.show()

    # get descent direction
    v_traj = ilqr_iter(x0, u_traj)

    # Armijo line search parameters
    gamma = 1.0  # initial step size
    alpha = 1e-04
    beta = 0.5

    total_loss = np.sum([loss(t*dt, x_traj[t], u_traj[t]) for t in range(tsteps)])
    total_other_loss = np.sum([loss(t*dt, x_traj[t], u_traj[t] + gamma * v_traj[t]) for t in range(tsteps)])
    loss_list.append(total_loss)

    print(f"Iteration {iter}: Loss = {total_loss}")
    print(f"Other Loss = {total_other_loss}")
    print(f"np.sum(v_traj * v_traj) = {np.sum(v_traj * v_traj)}")
    while total_other_loss > total_loss + alpha * gamma * np.sum(v_traj * v_traj):
        gamma = beta * gamma
        total_other_loss = np.sum([loss(t*dt, x_traj[t], u_traj[t] + gamma * v_traj[t]) for t in range(tsteps)])
        print(f"new other lsos: {total_other_loss}")

    # update control for the next iteration
    u_traj += gamma * v_traj

desired_traj = np.array([[2.0*t / np.pi, 0.0, np.pi/2.0] for t in range(tsteps)])
ax[0].plot(x_traj_initial[:,0], x_traj_initial[:,1], linestyle='--', color='k', label="Initial Trajectory")
ax[0].plot(desired_traj[:,0], desired_traj[:,1], linestyle='-', color='r', label="Desired Trajectory")
ax[0].plot(x_traj[:,0], x_traj[:,1], linestyle='-', color='k', label="Converged Trajectory")
ax[0].set_title('State Trajectory')
ax[0].legend(loc='upper right')
ax[0].set_xlim(0, 4)
ax[0].set_ylim(-2.25, 2.25)

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
ax[2].set_ylim(0, 2000)

plt.show()

# plt.plot(time, u_traj[:,0], label='u1')
# plt.plot(time, u_traj[:,1], label='u2')
# plt.legend()
# plt.xlim(0, 6)
# plt.ylim(-3, 3)
# plt.show()


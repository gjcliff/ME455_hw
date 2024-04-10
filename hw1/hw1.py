import numpy as np
import matplotlib.pyplot as plt

num_points = 100

#Problem 1
s1 = np.array((0.3, 0.4))
x1, x_pos, x_neg, y1, y_pos, y_neg, z1 = [], [], [], [], [], [], []

for i in range(num_points):
    xi, yi, u = np.random.uniform(size=3)
    zi = 0
    if u < np.exp(-100 * (np.linalg.norm(np.array((xi, yi)) - s1) - 0.2)**2):
        zi = 1
        x_pos.append(xi)
        y_pos.append(yi)
    else:
        zi = 0
        x_neg.append(xi)
        y_neg.append(yi)
    x1.append(xi)
    y1.append(yi)
    z1.append(zi)

def f(X,Y,s):
    Z = np.zeros((X.shape))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = np.exp(-100 * (np.linalg.norm((X[i,j], Y[i,j]) - s1) - 0.2)**2)
    return Z

# Convert z to colors
plt.ioff()
colors = ['red' if zi == 0 else 'green' for zi in z1]

x_points = np.linspace(0, 1, num_points)
y_points = np.linspace(0, 1, num_points)
X, Y = np.meshgrid(x_points, y_points)

# Plot the points
fig, axs = plt.subplots(1, 3, figsize=(15,4), facecolor='white')
fig.gca().set_facecolor('black')
fig.gca().set_aspect('equal', adjustable='box')
axs[0].contourf(X, Y, f(X,Y,s1), levels=6, cmap='gray')
axs[0].scatter(x_pos, y_pos, c='green', label='Positive Signal')
axs[0].scatter(x_neg, y_neg, c='red', label='Negative Signal')
axs[0].plot(s1[0], s1[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
axs[0].set_xlabel('y', fontsize=15)
axs[0].set_ylabel('x', fontsize=15)
axs[0].set_title('Problem 1', fontsize=20)
# fig.gca().set_facecolor('black')
# fig.gca().set_aspect('equal', adjustable='box')
axs[0].legend(loc='upper right')
display(fig)
# plt.show()

# Problem 2
def likelihood(Sx, Sy):
    likelihood = np.ones((Sx.shape))
    for i in range(Sx.shape[0]):
        for j in range(Sx.shape[1]):
            for k in range(len(z1)):
                if z1[k] == 1:
                    likelihood[i,j] *= np.exp(-100 * (np.linalg.norm(np.array((x1[k], y1[k])) - np.array((Sx[i,j], Sy[i,j]))) - 0.2)**2) 
                else:
                    likelihood[i,j] *= 1 - np.exp(-100 * (np.linalg.norm(np.array((x1[k], y1[k])) - np.array((Sx[i,j], Sy[i,j]))) - 0.2)**2)
    return likelihood

# using the same meshgrid as before since it doesn't change
# fig, ax = plt.subplots(facecolor='white')
axs[1].contourf(X, Y, likelihood(X,Y), levels=6, cmap='gray')
axs[1].scatter(x_pos, y_pos, c='green', label='Positive Signal')
axs[1].scatter(x_neg, y_neg, c='red', label='Negative Signal')
# axs[1].plot(s[0], s[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
axs[1].set_xlabel('y', fontsize=15)
axs[1].set_ylabel('x', fontsize=15)
axs[1].set_title('Problem 2', fontsize=20)
axs[1].legend(loc='upper right')
display(fig)

# Problem 3
def likelihood(Sx, Sy, x, y, z):
    likelihood = np.ones((Sx.shape))
    for i in range(Sx.shape[0]):
        for j in range(Sx.shape[1]):
            for k in range(len(z)):
                if z[k] == 1:
                    likelihood[i,j] *= np.exp(-100 * (np.linalg.norm(np.array((x, y)) - np.array((Sx[i,j], Sy[i,j]))) - 0.2)**2) 
                else:
                    likelihood[i,j] *= 1 - np.exp(-100 * (np.linalg.norm(np.array((x, y)) - np.array((Sx[i,j], Sy[i,j]))) - 0.2)**2)
    return likelihood

x3, y3 = np.random.uniform(size=2)
z3 = []
for i in range(num_points):
    z = np.random.uniform()
    if z < np.exp(-100 * (np.linalg.norm(np.array((x3, y3)) - s1) - 0.2)**2):
        z = 1
    else:
        z = 0
    z3.append(z)

# fig, ax = plt.subplots(facecolor='white')
axs[2].contourf(X, Y, likelihood(X,Y,x3,y3,z3), levels=6, cmap='gray')
axs[2].scatter(x_pos, y_pos, c='green', label='Positive Signal')
axs[2].scatter(x_neg, y_neg, c='red', label='Negative Signal')
axs[2].plot(x3, y3, color='purple', marker='o', markersize=5, markeredgewidth=5, label='Sensor')
axs[2].plot(s1[0], s1[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
axs[2].set_xlabel('y', fontsize=15)
axs[2].set_ylabel('x', fontsize=15)
axs[2].set_title('Problem 3', fontsize=20)
axs[2].legend(loc='upper right')
display(fig)

# Problem 4
fig4, axs4 = plt.subplots(5,2,figsize=(10,20),facecolor='white')
x4, y4 = np.random.uniform(size=2)
belief = np.full((num_points, num_points), 1/(num_points*num_points))
measurements = 10
for i in range(measurements):
    z = np.random.uniform()
    if z < np.exp(-100 * (np.linalg.norm(np.array((x4, y4)) - s1) - 0.2)**2):
        for j in range(X.shape[-1]):
            for k in range(X.shape[1]):
                belief[j,k] *= np.exp(-100 * (np.linalg.norm(np.array((x4, y4)) - np.array((X[j,k], Y[j,k]))) - 0.2)**2)
    else:
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                belief[j,k] *= 1 - np.exp(-100 * (np.linalg.norm(np.array((x4, y4)) - np.array((X[j,k], Y[j,k]))) - 0.2)**2)

    row = i // 2
    col = i % 2
    ax = axs4[row, col]
    ax.contourf(X, Y, belief, levels=6, cmap='gray')
    ax.scatter(x_pos, y_pos, c='green', label='Positive Signal')
    ax.scatter(x_neg, y_neg, c='red', label='Negative Signal')
    ax.plot(x4, y4, color='purple', marker='o', markersize=5, markeredgewidth=5, label='Sensor')
    ax.plot(s1[0], s1[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Problem 4 Iteration {i+1}")
    # ax.gca().set_facecolor('black')
    # ax.gca().set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')


plt.tight_layout()
plt.show()
    
# Problem 5
fig5, axs5 = plt.subplots(5,2,figsize=(10,20),facecolor='white')
belief = np.full((num_points, num_points), 1/(num_points*num_points))
measurements = 10
for i in range(measurements):
    x5, y5 = np.random.uniform(size=2)
    z = np.random.uniform()
    if z < np.exp(-100 * (np.linalg.norm(np.array((x5, y5)) - s1) - 0.2)**2):
        for j in range(X.shape[-1]):
            for k in range(X.shape[1]):
                belief[j,k] *= np.exp(-100 * (np.linalg.norm(np.array((x5, y5)) - np.array((X[j,k], Y[j,k]))) - 0.2)**2)
    else:
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                belief[j,k] *= 1 - np.exp(-100 * (np.linalg.norm(np.array((x5, y5)) - np.array((X[j,k], Y[j,k]))) - 0.2)**2)

    row = i // 2
    col = i % 2
    ax = axs5[row, col]
    ax.contourf(X, Y, belief, levels=6, cmap='gray')
    ax.scatter(x_pos, y_pos, c='green', label='Positive Signal')
    ax.scatter(x_neg, y_neg, c='red', label='Negative Signal')
    ax.plot(x5, y5, color='purple', marker='o', markersize=5, markeredgewidth=5, label='Sensor')
    ax.plot(s1[0], s1[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Problem 5 Iteration {i+1}")
    # ax.gca().set_facecolor 'black')
    # ax.gca().set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right')


plt.tight_layout()
plt.show()

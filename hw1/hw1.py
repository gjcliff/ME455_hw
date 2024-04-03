import numpy as np
import matplotlib.pyplot as plt

num_points = 100

#Problem 1
s = np.array((0.3, 0.4))
x, x_pos, x_neg, y, y_pos, y_neg, z = [], [], [], [], [], [], []

for i in range(num_points):
    xi, yi, u = np.random.uniform(size=3)
    zi = 0
    if u < np.exp(-100 * (np.linalg.norm(np.array((xi, yi)) - s) - 0.2)**2):
        zi = 1
        x_pos.append(xi)
        y_pos.append(yi)
    else:
        zi = 0
        x_neg.append(xi)
        y_neg.append(yi)
    x.append(xi)
    y.append(yi)
    z.append(zi)

def f(X,Y):
    Z = np.zeros((X.shape))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = np.exp(-100 * (np.linalg.norm((X[i,j], Y[i,j]) - s) - 0.2)**2)
    return Z

# Convert z to colors
colors = ['red' if zi == 0 else 'green' for zi in z]

x_points = np.linspace(0, 1, num_points)
y_points = np.linspace(0, 1, num_points)
X, Y = np.meshgrid(x_points, y_points)

# Plot the points
fig, ax = plt.subplots(facecolor='white')
plt.contourf(X, Y, f(X,Y), levels=6, cmap='gray')
plt.scatter(x_pos, y_pos, c='green', label='Positive Signal')
plt.scatter(x_neg, y_neg, c='red', label='Negative Signal')
plt.plot(s[0], s[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.gca().set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='upper right')
plt.show()

# Problem 2
def likelihood(Sx, Sy):
    likelihood = np.ones((Sx.shape))
    for i in range(Sx.shape[0]):
        for j in range(Sx.shape[1]):
            for k in range(len(z)):
                if z[k] == 1:
                    likelihood[i,j] *= np.exp(-100 * (np.linalg.norm(np.array((x[k], y[k])) - (Sx[i,j], Sy[i,j])) - 0.2)**2) 
                else:
                    likelihood[i,j] *= 1 - np.exp(-100 * (np.linalg.norm(np.array((x[k], y[k])) - s) - 0.2)**2)
    return likelihood

# using the same meshgrid as before since it doesn't change
fig, ax = plt.subplots(facecolor='white')
plt.contourf(X, Y, likelihood(X,Y), levels=6, cmap='gray')
plt.scatter(x_pos, y_pos, c='green', label='Positive Signal')
plt.scatter(x_neg, y_neg, c='red', label='Negative Signal')
# plt.plot(s[0], s[1], color='blue', marker='x', markersize=15, markeredgewidth=5, label='Source')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.gca().set_facecolor('black')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='upper right')
plt.show()


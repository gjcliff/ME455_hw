#|%%--%%| <CCLthl0oID|erXUkqlZ4y>

import numpy as np
import matplotlib.pyplot as plt
import itertools

from PIL import Image
from io import BytesIO
import requests

#|%%--%%| <erXUkqlZ4y|pdEazmmeZi>
r"""°°°
Problem 1
°°°"""
#|%%--%%| <pdEazmmeZi|eHLjSnlryM>

# URL of the image
image_url = 'https://raw.githubusercontent.com/MurpheyLab/ME455_public/main/figs/lincoln.jpg'

# Fetch the image data from the URL
response = requests.get(image_url)

# Create a BytesIO object from the response data
image_data = BytesIO(response.content)

# Open the image using PIL
image = Image.open(image_data)

# Convert the image to a NumPy array
image_array = np.array(image)
image_array = np.flip(image_array, axis=0)

print('image_array.shape: ', image_array.shape)

plt.imshow(image_array, origin='lower', cmap='gray') # note that for "imshow" the origin of the coordinate is at top left instead of bottom left
plt.show()
plt.close()

#|%%--%%| <eHLjSnlryM|8TEufM7Usf>

xgrids = np.linspace(0.0, 1.0, image_array.shape[0])  # the x coordinates of image pixels in the new space
dx = xgrids[1] - xgrids[0]
ygrids = np.linspace(0.0, 1.0, image_array.shape[1])  # the y coordinates of image pixels in the new space
dy = ygrids[1] - ygrids[0]

# we now invert dark and light pixel density and normalize the density values so it is a valid probability distribution
density_array = 255.0 - image_array  # we want higher density at darker regions
density_array /= np.sum(density_array) * dx * dy  # so the integral is 1

def image_density(s):
    """ Continuous density function based on the image
    Inputs:
        s - a numpy array containing the (x,y) coordinate within the 1m-by-1m space
    Return:
        val - the density value at s
    """
    s_x, s_y = s

    # Find the pixel closest to s in the 1-by-1 space
    # Note that in image the first pixel coordinate correspond to the y-axis in the 1-by-1 space
    pixel_idx_y = np.argmin(np.abs(xgrids - s_x))
    pixel_idx_x = np.argmin(np.abs(ygrids - s_y))

   # the density at s is the same as the closest pixel density
    val = density_array[pixel_idx_x, pixel_idx_y]

    return val

#|%%--%%| <8TEufM7Usf|GJLTw6Aw4A>
r"""°°°
Proposal distribution: Uniform
°°°"""
#|%%--%%| <GJLTw6Aw4A|i56dJ78999>

num_samples = 5000
gsamples = np.random.uniform(low=0.0, high=1.0, size=(num_samples, 2))
scalars = np.random.uniform(low=0.0, high=1.0, size=num_samples)
samples = np.zeros((num_samples,2))
weights = np.zeros((num_samples,))
M = 1.0

for i in range(num_samples):
    if image_density(gsamples[i]) > M * scalars[i]:
        samples[i] = gsamples[i]
        weights[i] = image_density(gsamples[i])

weights /= np.max(weights)

fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=120, tight_layout=True)
ax.set_aspect('equal')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)

for sample,weight in zip(samples, weights):
    ax.plot(sample[0], sample[1], linestyle='', marker='o', markersize=2, color='k', alpha=weight)

plt.title('Problem 1, Proposal distribution: Uniform')
plt.show()
plt.close()

#|%%--%%| <i56dJ78999|iQyjajUg0H>
r"""°°°
Proposal distribution: Gaussian
°°°"""
#|%%--%%| <iQyjajUg0H|7TgzxgtEWH>

num_samples = 5000
gsamples = np.random.normal(loc=0.5, scale=0.2, size=(num_samples, 2))
scalars = np.random.normal(loc=0.5, scale=0.1, size=num_samples)
samples = np.zeros((num_samples,2))
weights = np.zeros((num_samples,))
M = 1.0

for i in range(num_samples):
    if image_density(gsamples[i]) > M * scalars[i]:
        samples[i] = gsamples[i]
        weights[i] = image_density(gsamples[i])

weights /= np.max(weights)

fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=120, tight_layout=True)
ax.set_aspect('equal')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)

for sample,weight in zip(samples, weights):
    ax.plot(sample[0], sample[1], linestyle='', marker='o', markersize=2, color='k', alpha=weight)

plt.title('Problem 1, Proposal distribution: Gaussian')
plt.show()
plt.close()


#|%%--%%| <7TgzxgtEWH|NFEfULaatt>
r"""°°°
Problem 2
°°°"""
#|%%--%%| <yi9x4ev15p|VNUMRztYA1>

# Predefined list of colors
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Create a cycle iterator for colors
color_cycle = itertools.cycle(colors)

#|%%--%%| <NFEfULaatt|CPoZQKMctx>

def get_velocity(state, u):
    return [np.cos(state[2]) * u[0], np.sin(state[2]) * u[0], u[1]]

def get_measurement(state, sigma_measurement):
    copy_state = state.copy()
    return [copy_state[0] + np.random.normal(0, sigma_measurement),
            copy_state[1] + np.random.normal(0, sigma_measurement),
            copy_state[2] + np.random.normal(0, sigma_measurement)]

def take_samples(state, sigma_measurement):
    samples = []
    for i in range(num_samples):
        samples.append(get_measurement(state, sigma_measurement))
    return samples

def update_state(state, u, sigma_process):
    new_state = state.copy()
    new_state[0] += np.cos(state[2]) * u[0] * dt + np.random.normal(0, sigma_process)
    new_state[1] += np.sin(state[2]) * u[0] * dt + np.random.normal(0, sigma_process)
    new_state[2] += u[1] * dt + np.random.normal(0, sigma_process)
    return new_state

#|%%--%%| <CPoZQKMctx|cux8jzxeJi>
# start with taking a bunch of measurements from a gaussian distribution with mean
# 0 and sigma_measurement.
u = [1, -0.5]
state = [0, 0, np.pi/2] # x, y, theta
total_time = 2 * np.pi
dt = 0.1

num_samples = 100
sigma_process = 0.02
sigma_measurement = 0.2
time_steps = int(total_time / dt)

estimated_state = state

positions = [state[:2]]
estimate_positions = [state[:2]]
for t in range(time_steps):
    weights = np.ones((num_samples,)) / num_samples
    samples = take_samples(state, sigma_measurement)
    predictions = []
    for sample in samples:
        predictions.append(update_state(sample, u, sigma_process))
    # once we have our belief, we execute u
    state = update_state(state, u, sigma_process)
    positions.append(state)
    # now we receive a measurement z_{t+1}
    z = get_measurement(state, sigma_measurement)
    weights = 1/np.sqrt(2*np.pi*sigma_measurement**2) * np.exp(-0.5 * np.sum((np.array(z) - np.array(predictions))**2, axis=1) / sigma_measurement**2)
    normalized_weights = weights / np.sum(weights)
    estimated_state = np.sum(np.array(predictions) * normalized_weights[:, np.newaxis], axis=0)
    estimate_positions.append(estimated_state[:2])
    # resample
    resample_indices = np.random.choice(np.arange(num_samples), num_samples, p=normalized_weights)
    samples = np.array(samples)
    samples = samples[resample_indices]
    maximized_weights = weights / np.max(weights)
    plt.plot([position[0] for position in positions], [position[1] for position in positions], linestyle='-', marker='o', markersize=2, color='k')
    plt.plot([position[0] for position in estimate_positions], [position[1] for position in estimate_positions], linestyle='-', marker='o', markersize=2, color='r')
    if t % 10 == 0:
        color = next(color_cycle)
        for sample, weight in zip(samples, maximized_weights):
            plt.plot(sample[0], sample[1], linestyle='', marker='o', markersize=2, color=color, alpha=weight)  # Set alpha based on weight
        plt.xlim(-2, 5)
        plt.ylim(-2.5, 2.5)
    plt.draw()
    plt.pause(0.1)

#|%%--%%| <cux8jzxeJi|6HnUL3zXvI>
r"""°°°
Problem 3
°°°"""
#|%%--%%| <6HnUL3zXvI|K8teC0q2GS>
from scipy.stats import multivariate_normal

w1 = 0.5
w2 = 0.2
w3 = 0.3
weights = [w1, w2, w3]

K = 3
mew1 = np.array([0.35, 0.38])
mew2 = np.array([0.68, 0.25])
mew3 = np.array([0.56, 0.64])

sigma1 = np.array([[0.01, 0.004],[0.004, 0.01]])
sigma2 = np.array([[0.005, -0.003],[-0.003, 0.005]])
sigma3 = np.array([[0.008, 0.0], [0.0, 0.004]])

num_samples = 100

samples1 = []
samples2 = []
samples3 = []

for i in range(num_samples):
    choice = np.random.choice(np.arange(K), p=weights)
    if choice == 0:
        samples1.append(np.random.multivariate_normal(mew1, sigma1))
        print(samples1[-1])
    elif choice == 1:
        samples2.append(np.random.multivariate_normal(mew2, sigma2))
    else:
        samples3.append(np.random.multivariate_normal(mew3, sigma3))

samples1 = np.array(samples1)
samples2 = np.array(samples2)
samples3 = np.array(samples3)

plt.title("Unlabeled samnples")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(samples1[:,0], samples1[:,1], marker='o', edgecolor='k', facecolor='k', alpha=0.3)
plt.scatter(samples2[:,0], samples2[:,1], marker='o', edgecolor='k', facecolor='k', alpha=0.3)
plt.scatter(samples3[:,0], samples3[:,1], marker='o', edgecolor='k', facecolor='k', alpha=0.3)
plt.show()

#|%%--%%| <K8teC0q2GS|5Ia1JZlwc0>

plt.title("Ground Truth")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(samples1[:,0], samples1[:,1], marker='o', edgecolor='k', facecolor='r', alpha=0.3)
plt.scatter(samples2[:,0], samples2[:,1], marker='o', edgecolor='k', facecolor='g', alpha=0.3)
plt.scatter(samples3[:,0], samples3[:,1], marker='o', edgecolor='k', facecolor='b', alpha=0.3)
plt.show()

#|%%--%%| <5Ia1JZlwc0|tfqsw6Wvnh>

# expectation maximiazation

samples = np.concatenate([samples1, samples2, samples3], axis=0)

probablities = []

x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
pos = np.dstack((x, y)).reshape(-1, 2)

num_iters = 20
dummy_weights = np.ones((K,)) / K
dummy_mews = [np.random.uniform(0, 1, 2) for i in range(K)]
dummy_sigmas = [np.eye(2) for i in range(K)]
print(dummy_mews[0])
print(dummy_sigmas[0])
print(pos.shape)
fig, axs = plt.subplots(2, 2, figsize=(10, 20))
row, col = 0, 0
print(axs.shape)
# axs = axs.flatten()
for iter in range(num_iters):
    pdf1 = multivariate_normal.pdf(pos, mean=dummy_mews[0], cov=dummy_sigmas[0])
    pdf2 = multivariate_normal.pdf(pos, dummy_mews[1], dummy_sigmas[1])
    pdf3 = multivariate_normal.pdf(pos, dummy_mews[2], dummy_sigmas[2])
    pdf1 = pdf1.reshape(x.shape)
    pdf2 = pdf2.reshape(x.shape)
    pdf3 = pdf3.reshape(x.shape)

    # plt.title(f"EM Iteration {iter + 1}")
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.contourf(x, y, pdf1, cmap="Reds", alpha=0.6)
    # plt.contourf(x, y, pdf2, cmap="Greens", alpha=0.3)
    # plt.contourf(x, y, pdf3, cmap="Blues", alpha=0.3)
    # plt.scatter(samples[:,0], samples[:,1], marker='o', edgecolor='k', facecolor='k', alpha=0.3)
    # plt.draw()
    # plt.pause(0.1)
    if iter % 5 == 0:
        axs[row][col].contourf(x, y, pdf1.reshape(x.shape), cmap="Reds", alpha=0.6)
        axs[row][col].contourf(x, y, pdf2.reshape(x.shape), cmap="Greens", alpha=0.3)
        axs[row][col].contourf(x, y, pdf3.reshape(x.shape), cmap="Blues", alpha=0.3)
        axs[row][col].scatter(samples[:, 0], samples[:, 1], marker='o', edgecolor='k', facecolor='k', alpha=0.3)
        axs[row][col].set_xlim(0, 1)
        axs[row][col].set_ylim(0, 1)
        axs[row][col].set_title(f"EM Iteration {iter + 1}")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        if col == 1:
            col = 0
            row += 1
        else:
            col += 1

    gammas = np.zeros((num_samples, K))
    for i in range(len(samples)):
        for k in range(K):
            gammas[i][k] = (dummy_weights[k] * multivariate_normal(dummy_mews[k], dummy_sigmas[k]).pdf(samples[i]) /
                           np.sum([dummy_weights[j] * multivariate_normal(dummy_mews[j], dummy_sigmas[j]).pdf(samples[i]) for j in range(K)]))
    for k in range(K):
        Nk = np.sum(gammas[:, k])
        dummy_weights[k] = Nk / num_samples
        dummy_mews[k] = np.sum(gammas[:, k][:, np.newaxis] * samples, axis=0) / Nk
        dummy_sigmas[k] = np.sum([gammas[i][k] * np.outer(samples[i] - dummy_mews[k], samples[i] - dummy_mews[k]) for i in range(num_samples)], axis=0) / Nk
plt.show()
    


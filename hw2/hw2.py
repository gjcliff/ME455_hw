import numpy as np
import matplotlib.pyplot as plt
import pdb

#|%%--%%| <r6lGkSzWzd|zO08PorjKT>
def get_bayes():
    bayes = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                      [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                      [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                      [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    return bayes

def get_reverse_bayes():
    reverse_bayes = get_bayes()
    for i in range(reverse_bayes.shape[0]):
        for j in range(reverse_bayes.shape[1]):
            reverse_bayes[i,j] = 1 - reverse_bayes[i,j]
    return reverse_bayes

def get_likelihood(x, y, sensor_reading):
    try:
        if sensor_reading:
            likelihood = np.zeros((25, 25))
            bayes = get_bayes()
        else:
            likelihood = np.ones((25, 25))
            bayes = get_reverse_bayes()

        # Define the bounds for slicing
        y_start = np.clip(y - bayes.shape[0] // 2, 0, None)
        y_end = np.clip(y + bayes.shape[0] // 2 + 1, None, likelihood.shape[0])
        x_start = np.clip(x - bayes.shape[1] // 2, 0, None)
        x_end = np.clip(x + bayes.shape[1] // 2 + 1, None, likelihood.shape[1])

        # Perform slicing and update likelihood
        likelihood_slice = likelihood[y_start:y_end, x_start:x_end]
        bayes_slice = bayes[
            bayes.shape[0] // 2 - (y - y_start):bayes.shape[0] // 2 + (y_end - y),
            bayes.shape[1] // 2 - (x - x_start):bayes.shape[1] // 2 + (x_end - x),
        ]
        likelihood[y_start:y_end, x_start:x_end] = bayes_slice

        return likelihood

    except ValueError:
        breakpoint()

def new_bayes_update(belief, x, y, sensor_reading):
    likelihood = get_likelihood(x, y, sensor_reading)
    belief = np.multiply(belief, likelihood) / np.sum(np.multiply(belief, likelihood))
    return belief

#|%%--%%| <zO08PorjKT|gecaolf2fb>

def update_robot_position(belief, robot_x, robot_y, sensor_reading):
    max_prob = -100
    rv = np.random.randint(0,4)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    max_x = robot_x + directions[rv][0]
    max_y = robot_y + directions[rv][1]
    # if sensor_reading:
    #     breakpoint()
    for direction in directions:
        x = robot_x + direction[0]
        y = robot_y + direction[1]
        if x < 0 or x >= 25 or y < 0 or y >= 25:
            continue
        try:
            if belief[y, x] > max_prob:
                max_prob = belief[y, x]
                max_x = x
                max_y = y
        except IndexError:
            pass

    robot_x = max_x
    robot_y = max_y
    
    if robot_x < 0:
        robot_x = 0
    elif robot_x >= 25:
        robot_x = 24
    if robot_y < 0:
        robot_y = 0
    elif robot_y >= 25:
        robot_y = 24

    return robot_x, robot_y

def bernoulli_distribution(belief, x, y):
    for i in range(-3,4):
        try:
            if i == 0:
                belief[y, x] = 1.0
            if i < 2 and i > -2:
                if x + 1 < 25:
                    belief[y + 1, x + i] = 0.5
                if x - 1 >= 0:
                    belief[y - 1, x + i] = 0.5
            if i < 3 and i > -3:
                if x + 2 < 25:
                    belief[y + 2, x + i] = 0.33
                if x - 2 >= 0: 
                    belief[y - 2, x + i] = 0.33
            if i < 4 and i > -4:
                if x + 3 < 25:
                    belief[y + 3, x + i] = 0.25
                if x - 3 >= 0:
                    belief[y - 3, x + i] = 0.25
        except IndexError:
            pass
    return belief

#|%%--%%| <gecaolf2fb|eqe6nalgjz>

# Part 1

#pick a random starting point for the robot
robot_x = np.random.randint(0,25)
robot_y = np.random.randint(0,25)
prev_robot_positions = [(robot_x + 0.5, robot_y + 0.5)]
prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

# pick a random starting point for the food source
food_x = np.random.randint(0,25)
food_y = np.random.randint(0,25)

belief = np.full((25, 25), 1/(25**2))
simulator = np.zeros((25, 25))
simulator = bernoulli_distribution(simulator, food_x, food_y)

found = False
fig, axs = plt.subplots(2,5,figsize=(50,25))
fig.subplots_adjust(hspace=0.1, wspace=0.3)
fig.suptitle("Problem 1 3rd Pic")
row, col = 0, 0
for i in range(100):
    if robot_x == food_x and robot_y == food_y:
        belief[robot_y, robot_x] = 1.0

    # plt.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
    # plt.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
    # plt.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    # plt.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
    # # plt.colorbar(label="probability")
    # plt.title("infotaxis")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xticks(range(0, 26, 1), fontsize=7)
    # plt.yticks(range(0, 26, 1), fontsize=7)
    # plt.grid(True)
    # plt.pause(0.01)
    # plt.cla()

    if ((i + 1) % 10) == 0:
        ax = axs[row, col]
        ax.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
        ax.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
        ax.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
        ax.plot(robot_x + 0.5, robot_y + 0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
        ax.set_title(f"iteration {i + 1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks(range(0, 26, 1), fontsize=0.1)
        ax.set_yticks(range(0, 26, 1), fontsize=0.1)
        ax.tick_params(axis='both', which='major', labelsize=1)
        ax.grid(True)

        if row == 0:
            row = 1
        else:
            col += 1
            row = 0
    # first, select an action

    # next, simulate a measurement
    z = np.random.uniform(0, 1)
    if z < simulator[robot_y, robot_x]:
        belief = new_bayes_update(belief, robot_x, robot_y, True)

        robot_x, robot_y = update_robot_position(belief, robot_x, robot_y, True)   

        prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
        prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)
    else:
        belief = new_bayes_update(belief, robot_x, robot_y, False)

        robot_x, robot_y = update_robot_position(belief, robot_x, robot_y, False)   

        prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
        prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

plt.show()

#|%%--%%| <eqe6nalgjz|wXBtKx02zm>

def calc_entropy(belief):
    entropy = 0
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            if belief[i,j] != 0:
                entropy += -belief[i,j] * np.log(belief[i,j])

    return entropy

def infotaxis(robot_x, robot_y, belief):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

    entropies = []
    current_entropy = calc_entropy(belief)
    for direction in directions:
        belief_pos = new_bayes_update(np.copy(belief), robot_x + direction[0], robot_y + direction[1], True)
        belief_neg = new_bayes_update(np.copy(belief), robot_x + direction[0], robot_y + direction[1], False)

        entropy_pos = calc_entropy(belief_pos)
        entropy_neg = calc_entropy(belief_neg)
        p_pos = 0
        likelihood = get_likelihood(robot_x + direction[0], robot_y + direction[1], True)
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                p_pos += belief[i,j] * likelihood[i,j]
        p_neg = 1 - p_pos
        entropy = p_pos * (entropy_pos - current_entropy) + p_neg * (entropy_neg - current_entropy)
        entropies.append(entropy)
    
    # now I need to evaluate how good each of these beliefs are somehow, and I
    # can pick a direction based on that.
    min_entropy_change = 100
    index = 0
    for i in range(len(entropies)):
        if entropies[i] < min_entropy_change:
            min_entropy_change = entropies[i]
            index = i


    robot_x += directions[index][0]
    robot_y += directions[index][1]

    if robot_x < 0:
        robot_x = 0
    elif robot_x >= 25:
        robot_x = 24
    if robot_y < 0:
        robot_y = 0
    elif robot_y >= 25:
        robot_y = 24

    return robot_x, robot_y, current_entropy


#|%%--%%| <wXBtKx02zm|bcAmIlQ6H0>

# Part 2

#pick a random starting point for the robot
robot_x = np.random.randint(0,25)
robot_y = np.random.randint(0,25)
prev_robot_positions = [(robot_x + 0.5, robot_y + 0.5)]
prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

# pick a random starting point for the food source
food_x = np.random.randint(0,25)
food_y = np.random.randint(0,25)

belief = np.full((25, 25), 1/(25**2))
simulator = np.zeros((25, 25))
simulator = bernoulli_distribution(simulator, food_x, food_y)

found = False
entropy = 100

# data
beliefs = []
robot_positions = [(robot_x + 0.5, robot_y + 0.5)]
prev_x_coords_list = []
prev_y_coords_list = []

for i in range(100):
    beliefs.append(belief)
    robot_positions.append((robot_x + 0.5, robot_y + 0.5))
    prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
    prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)
    prev_x_coords_list.append(prev_robot_x_coords)
    prev_y_coords_list.append(prev_robot_y_coords)

    # check if the entropy is lower than the threshold, this means we probably
    # found the source
    if robot_x == food_x and robot_y == food_y:
        belief[robot_y, robot_x] = 1.0
    if entropy < 1e-3:
        break

    # plt.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
    # plt.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
    # plt.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    # plt.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
    # # plt.colorbar(label="probability")
    # plt.title("infotaxis")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.xticks(range(0, 26, 1), fontsize=7)
    # plt.yticks(range(0, 26, 1), fontsize=7)
    # plt.grid(True)
    # plt.pause(0.001)
    # plt.cla()

    # first, simulate a measurement
    z = np.random.uniform(0, 1)
    if z < simulator[robot_y, robot_x]:
        print("yes")
        robot_x, robot_y, entropy = infotaxis(robot_x, robot_y, belief)
        belief = new_bayes_update(belief, robot_x, robot_y, True)
    else:
        robot_x, robot_y, entropy = infotaxis(robot_x, robot_y, belief)
        belief = new_bayes_update(belief, robot_x, robot_y, False)

fig, axs = plt.subplots(2,5,figsize=(50, 25))
fig.subplots_adjust(hspace=0.1, wspace=0.3)
fig.suptitle("Problem 2 3rd Pic")
row, col = 0, 0
print(f"len beliefs: {len(beliefs)}, len_beliefs//10: {len(beliefs)//10}")
for i in range(len(beliefs)//10, len(beliefs)+1,len(beliefs)//10):
    print(f"i: {i}, row: {row}, col: {col}")
    ax = axs[row, col]
    ax.imshow(beliefs[i-1], cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
    ax.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
    ax.plot(prev_x_coords_list[i-1], prev_y_coords_list[i-1], color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    ax.plot(robot_positions[i][0], robot_positions[i][1], color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
    ax.set_title(f"iteration {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, 26, 1), fontsize=0.1)
    ax.set_yticks(range(0, 26, 1), fontsize=0.1)
    ax.tick_params(axis='both', which='major', labelsize=1)
    ax.grid(True)

    if row == 0:
        row = 1
    else:
        col += 1
        row = 0

plt.show()

#|%%--%%| <bcAmIlQ6H0|EZp8upf8JJ>

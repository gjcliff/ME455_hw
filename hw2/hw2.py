import numpy as np
import matplotlib.pyplot as plt
import pdb

#|%%--%%| <r6lGkSzWzd|GEcaOlF2FB>

tmp_pos = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                    [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                    [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                    [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])

def update_robot_position(belief, robot_x, robot_y):
    # this is the case that the robot stands still
    z = np.random.uniform(0, 1)
    if z < 0.2:
        return robot_x, robot_y

    max_prob = 0
    rv = np.random.randint(0,4)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    max_x = robot_x + directions[rv][0]
    max_y = robot_y + directions[rv][1]
    for direction in directions:
        x = robot_x + direction[0]
        y = robot_y + direction[1]
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

def bayes_update(belief, x, y, sensor_reading):
    eta = 1
    # for i in range(belief.shape[0]):
    #     for j in range(belief.shape[1]):
    copy_belief = np.copy(belief)
    if sensor_reading:
        for i in range(-3, 4):
            try:
                if i == 0:
                    belief[y, x] = 0.0
                if i < 2 and i > -2:
                    if 0 <= y + 1 < 25 and 0 <= x + i < 25:
                        if belief[y + 1, x + i] != 0:
                            belief[y + 1, x + i] *= 0.5 * eta
                    if 0 <= y - 1 < 25 and 0 <= x + i < 25:
                        if belief[y - 1, x + i] != 0:
                            belief[y - 1, x + i] *= 0.5 * eta
                if i < 3 and i > -3:
                    if 0 <= y + 2 < 25 and 0 <= x + i < 25:
                        if belief[y + 2, x + i] != 0:
                            belief[y + 2, x + i] *= (1 - 0.33) * eta
                    if 0 <= y - 2 < 25 and 0 <= x + i < 25:
                        if belief[y - 2, x + i] != 0:
                            belief[y - 2, x + i] *= (1 - 0.33) * eta
                if i < 4 and i > -4:
                    if 0 <= y + 3 < 25 and 0 <= x + i < 25:
                        if belief[y + 3, x + i] != 0:
                            belief[y + 3, x + i] *= (1 - 0.25) * eta
                    if 0 <= y - 3 < 25 and 0 <= x + i < 25:
                        if belief[y - 3, x + i] != 0: 
                            belief[y - 3, x + i] *= (1 - 0.25) * eta
            except IndexError:
                pass
    else:
        for i in range(-3, 4):
            try:
                if i == 0:
                    belief[y, x] = 0.0
                if i < 2 and i > -2:
                    if 0 <= y + 1 < 25 and 0 <= x + i < 25:
                        if belief[y + 1, x + i] != 0:
                            belief[y + 1, x + i] *= 0.5 * eta
                    if 0 <= y - 1 < 25 and 0 <= x + i < 25:
                        if belief[y - 1, x + i] != 0:
                            belief[y - 1, x + i] *= 0.5 * eta
                if i < 3 and i > -3:
                    if 0 <= y + 2 < 25 and 0 <= x + i < 25:
                        if belief[y + 2, x + i] != 0:
                            belief[y + 2, x + i] *= 0.33 * eta
                    if 0 <= y - 2 < 25 and 0 <= x + i < 25:
                        if belief[y - 2, x + i] != 0:
                            belief[y - 2, x + i] *= 0.33 * eta
                if i < 4 and i > -4:
                    if 0 <= y + 3 < 25 and 0 <= x + i < 25:
                        if belief[y + 3, x + i] != 0:
                            belief[y + 3, x + i] *= 0.25 * eta
                    if 0 <= y - 3 < 25 and 0 <= x + i < 25:
                        if belief[y - 3, x + i] != 0:
                            belief[y - 3, x + i] *= 0.25 * eta
            except IndexError:
                pass
        belief /= np.sum(copy_belief)
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
fig, axs = plt.subplots(5,2,figsize=(100,100))
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

    if (i % 10) == 0:
        ax = axs[row, col]
        ax.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
        ax.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
        ax.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
        ax.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
        ax.set_title(f"step {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks(range(0, 26, 1))
        ax.set_yticks(range(0, 26, 1))
        ax.grid(True)

        if not col:
            col = 1
        else:
            row += 1
            col = 0
    # first, select an action
    robot_x, robot_y = update_robot_position(belief, robot_x, robot_y)   

    prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
    prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

    # next, simulate a measurement
    z = np.random.uniform(0, 1)
    if z < simulator[robot_y, robot_x]:
        belief = bayes_update(belief, robot_x, robot_y, True)
    else:
        belief = bayes_update(belief, robot_x, robot_y, False)
plt.show()

#|%%--%%| <eqe6nalgjz|zO08PorjKT>
def get_bayes():
    bayes = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                        [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                        [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.33, 0.33, 0.33, 0.33, 0.33, 0.0],
                        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
    return bayes

def get_reverse_bayes():
    reverse_bayes = get_bayes()
    for i in range(reverse_bayes.shape[0]):
        for j in range(reverse_bayes.shape[1]):
            if reverse_bayes[i,j] != 0:
                reverse_bayes[i,j] = 1 - reverse_bayes[i,j]
    return reverse_bayes

def get_likelihood(x, y, sensor_reading):
    likelihood = np.zeros((25, 25))
    breakpoint()
    if sensor_reading:
        bayes = get_bayes()
        likelihood[y + -bayes.shape[0]//2 : y + bayes.shape[0]//2+1, x + -bayes.shape[1]//2 : x + bayes.shape[1]//2+1] = bayes
    else:
        reverse_bayes = get_reverse_bayes()
        likelihood[y + -reverse_bayes.shape[0]//2 : y + reverse_bayes.shape[0]//2+1,
                   x + -reverse_bayes.shape[1]//2 : x + reverse_bayes.shape[1]//2+1] = reverse_bayes
    return likelihood

def new_bayes_update(belief, x, y, sensor_reading):
    likelihood = get_likelihood(x, y, sensor_reading)
    belief = np.multiply(belief, likelihood) / np.sum(np.multiply(belief, likelihood))
    return belief

#|%%--%%| <zO08PorjKT|wXBtKx02zm>

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
        # breakpoint()
        belief_pos = new_bayes_update(np.copy(belief), robot_x + direction[0], robot_y + direction[1], True)
        belief_neg = new_bayes_update(np.copy(belief), robot_x + direction[0], robot_y + direction[1], False)

        entropy_pos = calc_entropy(belief_pos)
        entropy_neg = calc_entropy(belief_neg)
        if robot_y - 3 < 0:
            if robot_x - 3 < 0:
                tmp_belief_pos = belief[robot_y:robot_y+4, robot_x:robot_x+4]
            elif robot_x + 3 >= 25:
                tmp_belief_pos = belief[robot_y:robot_y+4, robot_x-3:]
            else:
                tmp_belief_pos = belief[robot_y:robot_y+4, robot_x-3:robot_x+4]
        elif robot_y + 3 >= 25:
            if robot_x - 3 < 0:
                tmp_belief_pos = belief[robot_y-3:, robot_x:robot_x+4]
            elif robot_x + 3 >= 25:
                tmp_belief_pos = belief[robot_y-3:, robot_x-3:]
            else:
                tmp_belief_pos = belief[robot_y-3:, robot_x-3:robot_x+4]
        else:
            if robot_x - 3 < 0:
                tmp_belief_pos = belief[robot_y-3:robot_y+4, robot_x:robot_x+4]
            elif robot_x + 3 >= 25:
                tmp_belief_pos = belief[robot_y-3:robot_y+4, robot_x-3:]
            else:
                tmp_belief_pos = belief[robot_y-3:robot_y+4, robot_x-3:robot_x+4]
                tmp_belief_pos = belief[robot_y-3:robot_y+4, robot_x-3:robot_x+4]
        p_pos = np.sum(np.multiply(tmp_belief_pos, tmp_pos[:tmp_belief_pos.shape[0], :tmp_belief_pos.shape[1]]))
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


#|%%--%%| <NJX6vwV53I|bcAmIlQ6H0>

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
# fig, axs = plt.subplots(5,2,figsize=(100,100))
row, col = 0, 0
entropy = 100
for i in range(150):
    if robot_x == food_x and robot_y == food_y:
        belief[robot_y, robot_x] = 1.0
    if entropy < 1e-3:
        break

    plt.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
    plt.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
    plt.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    plt.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
    # plt.colorbar(label="probability")
    plt.title("infotaxis")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(range(0, 26, 1), fontsize=7)
    plt.yticks(range(0, 26, 1), fontsize=7)
    plt.grid(True)
    plt.pause(0.001)
    plt.cla()

    # if (i % 10) == 0:
    #     ax = axs[row, col]
    #     ax.imshow(belief, cmap='viridis', origin='lower', extent=[0, 25, 0, 25])
    #     ax.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='source')
    #     ax.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    #     ax.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='robot')
    #     ax.set_title(f"step {i}")
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_xticks(range(0, 26, 1))
    #     ax.set_yticks(range(0, 26, 1))
    #     ax.grid(True)
    #
    #     if not col:
    #         col = 1
    #     else:
    #         row += 1
    #         col = 0
    # first, simulate a measurement
    z = np.random.uniform(0, 1)
    if z < simulator[robot_y, robot_x]:
        print("yes")
        # breakpoint()
        belief = new_bayes_update(belief, robot_x, robot_y, True)
        robot_x, robot_y, entropy = infotaxis(robot_x, robot_y, belief)
    else:
        belief = new_bayes_update(belief, robot_x, robot_y, False)
        robot_x, robot_y, entropy = infotaxis(robot_x, robot_y, belief)

    prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
    prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

# plt.show()

#|%%--%%| <bcAmIlQ6H0|EZp8upf8JJ>

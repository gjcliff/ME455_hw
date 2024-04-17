import numpy as np
import matplotlib.pyplot as plt

#|%%--%%| <r6lGkSzWzd|GEcaOlF2FB>

def update_robot_position(belief, robot_x, robot_y):
    # this is the case that the robot stands still
    z = np.random.uniform(0, 1)
    if z < 0.2:
        return robot_x, robot_y

    max_prob = 0
    max_x = robot_x
    max_y = robot_y
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
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

def update_belief(belief, x, y, sensor_reading):
    if not sensor_reading:
        for i in range(-3,4):
            try:
                if i == 0:
                    belief[y, x] = 0.0
                if i < 2 and i > -2:
                    if x + 1 < 25:
                        belief[y + 1, x + i] *= 0.5
                    if x - 1 >= 0:
                        belief[y - 1, x + i] *= 0.5
                if i < 3 and i > -3:
                    if x + 2 < 25:
                        belief[y + 2, x + i] *= 1 - 0.33
                    if x - 2 >= 0: 
                        belief[y - 2, x + i] *= 1 - 0.33
                if i < 4 and i > -4:
                    if x + 3 < 25:
                        belief[y + 3, x + i] *= 1 - 0.25
                    if x - 3 >= 0:
                        belief[y - 3, x + i] *= 1 - 0.25
            except IndexError:
                pass
    else:
        for i in range(-3,4):
            try:
                if i == 0:
                    belief[y, x] = 0.0
                if i < 2 and i > -2:
                    if x + 1 < 25:
                        belief[y + 1, x + i] *= 0.5
                    if x - 1 >= 0:
                        belief[y - 1, x + i] *= 0.5
                if i < 3 and i > -3:
                    if x + 2 < 25:
                        belief[y + 2, x + i] *= 0.33
                    if x - 2 >= 0: 
                        belief[y - 2, x + i] *= 0.33
                if i < 4 and i > -4:
                    if x + 3 < 25:
                        belief[y + 3, x + i] *= 0.25
                    if x - 3 >= 0:
                        belief[y - 3, x + i] *= 0.25
            except IndexError:
                pass

    return belief

#|%%--%%| <gecaolf2fb|eqe6nalgjz>

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
    # plt.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='Source')
    # plt.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
    # plt.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='Robot')
    # # plt.colorbar(label="Probability")
    # plt.title("Infotaxis")
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
        ax.plot(food_x+0.5, food_y+0.5, color='green', marker='.', markersize=3, markeredgewidth=3, label='Source')
        ax.plot(prev_robot_x_coords, prev_robot_y_coords, color='blue', linestyle='-', marker='.', markersize=3, markeredgewidth=3)
        ax.plot(robot_x+0.5, robot_y+0.5, color='red', marker='.', markersize=3, markeredgewidth=3, label='Robot')
        ax.set_title(f"Step {i}")
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

    # first, select and action
    robot_x, robot_y = update_robot_position(belief, robot_x, robot_y)   

    prev_robot_positions.append((robot_x + 0.5, robot_y + 0.5))
    prev_robot_x_coords, prev_robot_y_coords = zip(*prev_robot_positions)

    # next, simulate a measurement
    z = np.random.uniform(0, 1)
    if z < simulator[robot_y, robot_x]:
        belief = update_belief(belief, robot_x, robot_y, True)
    else:
        belief = update_belief(belief, robot_x, robot_y, False)

plt.show()
#|%%--%%| <EQe6NALgJz|wXBtKx02zm>

def infotaxis(belief, robot_x, robot_y, sensor_reading):
    robot_x, robot_y = update_robot_position(belief, robot_x, robot_y)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

    

#|%%--%%| <wXBtKx02zm|EZp8upf8JJ>

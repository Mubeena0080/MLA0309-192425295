import numpy as np
import random

# Grid parameters
GRID_SIZE = 5
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = len(ACTIONS)

# Q-learning parameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 1000

# Rewards
FOOD_REWARD = 10
GHOST_PENALTY = -10
STEP_PENALTY = -1

# Positions
FOOD_POS = (4, 4)
GHOST_POS = (2, 2)

# Initialize Q-table
Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

# Environment step function
def step(state, action):
    x, y = state

    if action == 0 and x > 0:            # UP
        x -= 1
    elif action == 1 and x < GRID_SIZE - 1:  # DOWN
        x += 1
    elif action == 2 and y > 0:          # LEFT
        y -= 1
    elif action == 3 and y < GRID_SIZE - 1:  # RIGHT
        y += 1

    new_state = (x, y)

    if new_state == FOOD_POS:
        return new_state, FOOD_REWARD, True
    elif new_state == GHOST_POS:
        return new_state, GHOST_PENALTY, True
    else:
        return new_state, STEP_PENALTY, False

# Training
for episode in range(episodes):
    state = (0, 0)  # start position
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state, reward, done = step(state, action)

        best_next_action = np.max(Q[next_state[0], next_state[1]])
        Q[state[0], state[1], action] += alpha * (
            reward + gamma * best_next_action - Q[state[0], state[1], action]
        )

        state = next_state

print("Training completed!")

# Evaluation (Greedy policy)
state = (0, 0)
path = [state]
done = False

while not done:
    action = np.argmax(Q[state[0], state[1]])
    state, reward, done = step(state, action)
    path.append(state)

print("\nOptimal Path Learned:")
print(path)

import random

# Grid size
GRID = 5

# Goal position
GOAL = (4, 4)

# Actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Number of robots
robots = 2

# Initialize Q-tables
Q = [{} for _ in range(robots)]

# Move function
def move(pos, action):
    x, y = pos
    if action == 'UP' and x > 0:
        x -= 1
    elif action == 'DOWN' and x < GRID - 1:
        x += 1
    elif action == 'LEFT' and y > 0:
        y -= 1
    elif action == 'RIGHT' and y < GRID - 1:
        y += 1
    return (x, y)

episodes = 100
success = 0
alpha = 0.1  # learning rate

for ep in range(episodes):
    positions = [(0, 0), (0, 1)]

    for step in range(20):
        states = []
        actions = []

        # Store states and choose actions
        for i in range(robots):
            state = positions[i]
            states.append(state)

            if state not in Q[i]:
                Q[i][state] = {a: 0 for a in ACTIONS}

            action = random.choice(ACTIONS)
            actions.append(action)

        # Move robots
        new_positions = []
        for i in range(robots):
            new_positions.append(move(positions[i], actions[i]))

        positions = new_positions

        # Shared reward
        if all(p == GOAL for p in positions):
            reward = 10
            success += 1
        else:
            reward = -1

        # Update Q-values safely
        for i in range(robots):
            Q[i][states[i]][actions[i]] += alpha * reward

        if reward == 10:
            break

print("Successful missions:", success)

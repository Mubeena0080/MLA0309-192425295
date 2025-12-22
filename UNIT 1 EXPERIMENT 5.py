import numpy as np

states = [0, 1, 2]          # locations
actions = [-1, 1]          # left, right
pickup = 2
gamma = 0.9

V = np.zeros(len(states))

for _ in range(50):        # value iteration
    newV = V.copy()
    for s in states:
        values = []
        for a in actions:
            ns = min(max(s + a, 0), 2)
            reward = 10 if ns == pickup else -1
            values.append(reward + gamma * V[ns])
        newV[s] = max(values)
    V = newV

# optimal policy
policy = {}
for s in states:
    policy[s] = max(actions, key=lambda a: 
        (10 if min(max(s+a,0),2)==pickup else -1) + gamma*V[min(max(s+a,0),2)])

print("Optimal Value:", V)
print("Optimal Policy:", policy)

import random
import math

# -----------------------------
# Smart Grid Environment
# -----------------------------
class SmartGridEnv:
    def _init_(self):
        self.reset()

    def reset(self):
        self.demand = random.randint(5, 15)
        self.renewable = random.randint(0, 10)
        self.price = random.randint(3, 8)
        self.storage = 5
        return self.get_state()

    def get_state(self):
        return (self.demand, self.renewable, self.price, self.storage)

    def step(self, action):
        cost = 0

        if action == 0:      # use grid
            cost = self.demand * self.price
        elif action == 1:    # use renewable
            used = min(self.demand, self.renewable)
            cost = (self.demand - used) * self.price
        elif action == 2:    # store energy
            store = min(self.renewable, 3)
            self.storage += store
            cost = self.demand * self.price
        elif action == 3:    # release storage
            release = min(self.storage, self.demand)
            self.storage -= release
            cost = (self.demand - release) * self.price

        reward = -cost
        return self.reset(), reward

# -----------------------------
# Policy (No libraries)
# -----------------------------
class Policy:
    def _init_(self):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(4)]

    def softmax(self, vals):
        exps = [math.exp(v) for v in vals]
        s = sum(exps)
        return [v / s for v in exps]

    def act(self, state):
        logits = [self.weights[i] * state[i] for i in range(4)]
        probs = self.softmax(logits)
        return self.sample(probs), probs

    def sample(self, probs):
        r = random.random()
        total = 0
        for i, p in enumerate(probs):
            total += p
            if r < total:
                return i
        return 3

# -----------------------------
# KL Divergence
# -----------------------------
def kl_div(p_old, p_new):
    kl = 0
    for i in range(len(p_old)):
        if p_old[i] > 0:
            kl += p_old[i] * math.log(p_old[i] / p_new[i])
    return kl

# -----------------------------
# TRPO Training (5 episodes)
# -----------------------------
env = SmartGridEnv()
policy = Policy()

episodes = 5
lr = 0.05
max_kl = 0.01

for ep in range(episodes):
    state = env.reset()
    action, old_probs = policy.act(state)
    next_state, reward = env.step(action)

    advantage = reward
    new_weights = policy.weights[:]

    for i in range(4):
        new_weights[i] += lr * advantage

    old_logits = [policy.weights[i] * state[i] for i in range(4)]
    new_logits = [new_weights[i] * state[i] for i in range(4)]

    old_p = policy.softmax(old_logits)
    new_p = policy.softmax(new_logits)

    kl = kl_div(old_p, new_p)

    if kl < max_kl:
        policy.weights = new_weights

    print("Episode:", ep + 1,
          "Reward:", reward,
          "KL:", round(kl, 5),
          "Weights:", [round(w, 3) for w in policy.weights])

print("Done")

import gym
import numpy as np

env = gym.make("MountainCar-v0")

# Q-Learning Settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 1000

# Exploration Settings
epsilon = 1 # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:
            #Maximum possible Q value in next step(for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            #Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            #And here is equation for a new Q value for current state action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            #Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achieved - update Q value reward directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

        # Decaying is being done episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

env.close()
import gym
import numpy as np


env = gym.make("MountainCar-v0")
# env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)
learning_rate = 0.1

#measure of how much we values future reward over current reward(0,1)
discount = 0.95 

episodes = 25000 
show = 500

epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episodes//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


d_os_size = [20] * len(env.observation_space.high)
d_os_win_size = (env.observation_space.high - env.observation_space.low) / d_os_size
# print(d_os_size)
# print(d_os_win_size)

qtable = np.random.uniform(low=-2, high=0, size=(d_os_size + [env.action_space.n]))
# print(qtable)

def get_discrete_state(state):
    d_state = (state - env.observation_space.low)/ d_os_win_size
    return tuple(d_state.astype(np.int))



for episode in range(episodes):
    discrete_state = get_discrete_state(env.reset())
    done = False
    

    if episode % show == 0:
        print(episode)
        render = True
    else:
        render = False
    

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(qtable[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        action = np.argmax(qtable[discrete_state])
        new_state, reward, done, _ = env.step(action)

        new_d_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(qtable[new_d_state])
            current_q =qtable[discrete_state + (action, )]

            new_q =(1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            qtable[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Made it on episode {episode}")
            qtable[discrete_state + (action, )] = 0

        discrete_state = new_d_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
env.close()




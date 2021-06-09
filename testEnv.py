import gym
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

env = gym.make("CartPole-v1" )
# env = gym.make('BipedalWalker-v3')
# env.reset()


# print(env.observation_space.high)

learning_rate = 0.1

#measure of how much we values future reward over current reward(0,1)
discount = 0.95 

episodes = 2000
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


ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg':[], 'min':[], 'max':[]}


def get_discrete_state(state):
    d_state = (state - env.observation_space.low)/ d_os_win_size
    return tuple(d_state.astype(np.int))



for episode in range(episodes):
    episode_reward = 0
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
        episode_reward += reward

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
    
    ep_rewards.append(episode_reward)

    if not episode % show:
        average_reward = sum(ep_rewards[-show:]) /len(ep_rewards[-show:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-show:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-show:]))

        print(f"Episode : {episode} , avf: {average_reward}, min: {min(ep_rewards[-show:])}, MAX: {max(ep_rewards[-show:])}")


env.close()
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()





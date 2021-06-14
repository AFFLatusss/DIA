import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd

# pos_space = np.linspace(-1.2, 0.6, 20)
# vel_space = np.linspace(-0.07, 0.07, 20)

def to_discrete(pos_limit, vel_limit, bin_size=20):

    pos_discrete = np.linspace(pos_limit[0], pos_limit[1], bin_size)
    vel_discrete = np.linspace(vel_limit[0], vel_limit[1], bin_size)

    return pos_discrete, vel_discrete

def get_state(observation, pos_space, vel_space):
    pos, vel = observation
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)

    return(pos_bin, vel_bin)


def max_action(Q, state, actions=[0,1,2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

def create_state_space(bin_size=20):
    state_space = []
    for pos in range(bin_size):
        for vel in range(bin_size):
            state_space.append((pos,vel))

    return state_space

def get_env_limit(env):
    pos_high = env.observation_space.high[0]
    pos_low =  env.observation_space.low[0]

    vel_high = env.observation_space.high[1]
    vel_low =  env.observation_space.low[1]

    return (pos_high,pos_low), (vel_high,vel_low)

def start_q(epsilon,Q_score,Q_total_rewards):
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs, pos_space, vel_space)
        if i % 1000 == 0 and i > 0:
            print(f'episode {i}, score {Q_score}, epsilon {epsilon:.3f}')

        Q_score = 0
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < epsilon\
                else max_action(Q, state)
            new_obs, reward, done, info = env.step(action)
            new_state = get_state(new_obs, pos_space, vel_space)
            # print(new_obs)
            Q_score += reward

            action_ = max_action(Q, new_state)
            # ! update Q table with new value
            Q[state, action] = Q[state, action] + \
                learning_rate*(reward + gamma*Q[new_state, action_] - Q[state, action])
            state = new_state

            # if new_obs[0] >= env.goal_position:
                # print(done)
                # print(f"done on episode{i} ")

        Q_total_rewards[i] = Q_score
        # *Epsilon decay
        epsilon = epsilon - 2/n_games if epsilon > 0.01 else 0.01

    # *Plot graphs
    mean_rewards = np.zeros(n_games)
    max_rewards = np.zeros(n_games)
    min_rewards = np.zeros(n_games)
    num_game = np.zeros(n_games)
    for t in range(n_games):
        num_game[t] = t
        mean_rewards[t] = np.mean(Q_total_rewards[max(0,t-50):(t+1)])
        max_rewards[t] = np.max(Q_total_rewards[max(0,t-50):(t+1)])
        min_rewards[t] = np.min(Q_total_rewards[max(0,t-50):(t+1)])

    # plt.plot(mean_rewards)
    # plt.savefig('Q-Learning_rewards.png')

    q_df = pd.DataFrame({'num_games':num_game, 'mean_rewards':mean_rewards, 'max_rewards':max_rewards, 'min_rewards':min_rewards})

    return q_df
    




def start_sarsa(epsilon,S_score,S_total_rewards):
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs, pos_space, vel_space)
        if i % 1000 == 0 and i > 0:
            print(f'episode {i}, score {S_score}, epsilon {epsilon:.3f}')

        S_score = 0
        action = max_action(S, state) if np.random.random() > epsilon else env.action_space.sample()

        while not done:
        #     if i % 1000 == 0:
        #         env.render()
        
            # action = np.random.choice([0,1,2]) if np.random.random() < epsilon\
            #     else max_action(Q, state)
            new_obs, reward, done, info = env.step(action)
            new_state = get_state(new_obs, pos_space, vel_space)
            # print(new_obs)
            S_score += reward

            action_ = max_action(S, new_state)
            # ! update Q table with new value
            S[state, action] = S[state, action] + \
                learning_rate*(reward + gamma*S[new_state, action_] - S[state, action])
            state = new_state
            action = action_

            # if new_obs[0] >= env.goal_position:
                # print(done)
                # print(f"done on episode{i} ")

        S_total_rewards[i] = S_score
        # *Epsilon decay
        epsilon = epsilon - 2/n_games if epsilon > 0.01 else 0.01

    # *Plot graphs
    mean_rewards = np.zeros(n_games)
    max_rewards = np.zeros(n_games)
    min_rewards = np.zeros(n_games)
    num_game = np.zeros(n_games)
    for t in range(n_games):
        num_game[t] = t
        mean_rewards[t] = np.mean(S_total_rewards[max(0,t-50):(t+1)])
        max_rewards[t] = np.max(S_total_rewards[max(0,t-50):(t+1)])
        min_rewards[t] = np.min(S_total_rewards[max(0,t-50):(t+1)])

    s_df = pd.DataFrame({'num_games':num_game, 'mean_rewards':mean_rewards,'max_rewards':max_rewards, 'min_rewards':min_rewards })

    return s_df
    


if __name__ == "__main__":
    # ? initial setup
    plt.style.use('ggplot')
    env = gym.make("MountainCar-v0")

    #The number of bin we used to convert continuous space to discrete space
    bin_size = 30 
    pos_limit, vel_limit = get_env_limit(env)
    pos_space, vel_space = to_discrete(pos_limit, vel_limit, bin_size)



    n_games = 25000
    learning_rate = 0.1
    gamma = 0.99
    epsilon =  1.0
    env._max_episode_steps = 1000

#   ! Generate all possible state
    states = create_state_space(bin_size)

#  ! Create Q table for both methods
    Q = {}
    S = {}
    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0
            S[state,action] = 0 

    Q_score = 0
    Q_total_rewards = np.zeros(n_games)
    
    S_score = 0
    S_total_rewards = np.zeros(n_games)


    q_data = start_q(epsilon,Q_score,Q_total_rewards )
    print("Starting Sarsa")

    #* Reset epsilon for Sarsa
    epsilon =  1.0
    s_data = start_sarsa(epsilon, S_score, S_total_rewards)

    q_lines = q_data.plot.line(x='num_games')
    q_lines.figure.savefig('Q-Learning_rewards.png')

    s_lines = s_data.plot.line(x='num_games')
    s_lines.figure.savefig('Sarsa_rewards.png')


    
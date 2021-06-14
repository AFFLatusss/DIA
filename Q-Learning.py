import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation


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

def save_frames_as_gif(frames, filename, path='./', filetype='.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + str(filename) + filetype, writer='imagemagick', fps=60)

if __name__ == "__main__":
    # ? initial setup
    plt.style.use('ggplot')
    env = gym.make("MountainCar-v0")

    #The number of bin we used to convert continuous space to discrete space
    bin_size = 30 
    pos_limit, vel_limit = get_env_limit(env)
    pos_space, vel_space = to_discrete(pos_limit, vel_limit, bin_size)



    n_games = 50000
    learning_rate = 0.1
    gamma = 0.99
    epsilon =  1.0
    env._max_episode_steps = 2000

    states = create_state_space(bin_size)

    Q = {}
    for state in states:
        for action in [0,1,2]:
            Q[state,action] = 0


    score = 0
    total_rewards = np.zeros(n_games)
    aggr_ep_rewards = {"episode":[], "average":[], "minimum":[], "maximum":[]}


    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs, pos_space, vel_space)
        if i % 1000 == 0 and i > 0:
            print(f'episode {i}, score {score}, epsilon {epsilon:.3f}')

        score = 0
        frames = []
        while not done:
            if i % 10000 == 0 :
                frames.append(env.render(mode="rgb_array"))
            elif frames:
                save_frames_as_gif(frames, i)
            else:
                frames = []

            action = np.random.choice([0,1,2]) if np.random.random() < epsilon\
                else max_action(Q, state)
            new_obs, reward, done, info = env.step(action)
            new_state = get_state(new_obs, pos_space, vel_space)
            # print(new_obs)
            score += reward

            action_ = max_action(Q, new_state)
            # ! update Q table with new value
            Q[state, action] = Q[state, action] + \
                learning_rate*(reward + gamma*Q[new_state, action_] - Q[state, action])
            state = new_state

            # if new_obs[0] >= env.goal_position:
                # print(done)
                # print(f"done on episode{i} ")

        total_rewards[i] = score
        # *Epsilon decay
        epsilon = epsilon - 2/n_games if epsilon > 0.01 else 0.01

    # *Plot graphs
    mean_rewards = np.zeros(n_games)
    max_rewards = np.zeros(n_games)
    min_rewards = np.zeros(n_games)
    num_game = np.zeros(n_games)
    for t in range(n_games):
        num_game[t] = t
        mean_rewards[t] = np.mean(total_rewards[max(0,t-50):(t+1)])
        max_rewards[t] = np.max(total_rewards[max(0,t-50):(t+1)])
        min_rewards[t] = np.min(total_rewards[max(0,t-50):(t+1)])

    # plt.plot(mean_rewards)
    # plt.savefig('Q-MountainCar.png')

    plt.plot(num_game, mean_rewards, label="avg")
    plt.plot(num_game, min_rewards, label="min")
    plt.plot(num_game, max_rewards, label="max")
    plt.legend(loc=2)
    plt.show()

        

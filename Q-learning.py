import gym
import random
import numpy as np

env = gym.make("CartPole-v1" )

# Show possible action to the agent
print(env.action_space)

#Show observation space
print(env.observation_space)

# show action meaning
# print(env.unwrapped.get_action_meanings())
env.reset()
done = False


print(env.observation_space.high)
print(env.observation_space.low)

# while not done:
#     action = random.randint(0, 3) # always go right!
#     new_state, reward, done, _ = env.step(action)
#     print(reward, new_state)



for i_episode in range(5):
    observation = env.reset()
    print(1)
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        # print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
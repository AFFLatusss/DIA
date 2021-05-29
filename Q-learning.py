import gym
import random

env = gym.make('Breakout-v0')
# Show possible action to the agent
print(env.action_space.n)

#Show observation space
print(env.observation_space)

# show action meaning
print(env.unwrapped.get_action_meanings())
env.reset()
done = False

while not done:
    action = random.randint(0, 2) # always go right!
    new_state, reward, done, _ = env.step(action)
    # print(reward, new_state)



# for i_episode in range(5):
#     observation = env.reset()
#     print(1)
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         # print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
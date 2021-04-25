import gym

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

d_os_size = [20] * len(env.observation_space.high)
d_os_win_size = (env.observation_space.high - env.observation_space.low) / d_os_size
print(d_os_size)
print(d_os_win_size)


# done = False

# while not done:
#     action = 2
#     new_state, reward, done, _ = env.step(action)
#     print(new_state)
#     env.render()

# env.close()
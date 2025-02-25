import gymnasium as gym
import numpy as np
import matplotlib.pylab as plt
from DQN import DQN
from tqdm import tqdm

env = gym.make('CartPole-v1')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
max_steps = 300
num_episodes = 5000

model = DQN(state_space, action_space, hidden_size=64, batch_size=64, target_update_freq=10000)

rewards = []
epochs = []
td_errors = []
epsilons = []
for ep in tqdm(range(num_episodes)):
    state, info = env.reset()
    model.epsilon_update()
    ep_loss = 0
    ep_reward = 0
    for step in range(max_steps):
        action = model.predict(state)
        next_state, reward, _, done, info = env.step(action)
        model.push(state, action, next_state, reward, done)
        loss = model.backward()
        ep_reward += reward
        ep_loss += 0 if loss is None else loss
        state = next_state
        if done:
            break
    epsilons.append(model.eps)
    td_errors.append(ep_loss)
    epochs.append(step)
    rewards.append(ep_reward)



fg, axis = plt.subplots(3, 1, layout='constrained')

axis[0].plot(range(num_episodes), rewards)
axis[0].set_xlabel("number of episodes")
axis[0].set_ylabel("Reward per Episodes")

axis[1].plot(range(num_episodes), td_errors)
axis[1].set_xlabel("number of episodes")
axis[1].set_ylabel("Loss")

axis[2].plot(range(num_episodes), epsilons)
axis[2].set_xlabel("number of episodes")
axis[2].set_ylabel("epsilon")

plt.show()
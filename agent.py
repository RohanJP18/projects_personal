import numpy as np
np.bool8 = np.bool_
import gym
import random

def main():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    alpha, gamma, epsilon = 0.8, 0.95, 0.1
    episodes, max_steps = 2000, 100
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        for _ in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            result = env.step(action)
            if len(result) == 5:
                new_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                new_state, reward, done, _ = result
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            state = new_state
            if done:
                break

    wins = 0
    tests = 100
    for _ in range(tests):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            result = env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, reward, done, _ = result
        if reward == 1:
            wins += 1

    print(f"Win rate after training: {wins/tests:.2f}")

if __name__ == "__main__":
    main()

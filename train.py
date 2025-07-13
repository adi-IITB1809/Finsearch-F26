import gymnasium as gym
import torch
from agent import DQNAgent

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(state_dim, action_dim, device=device)

    num_episodes = 500
    max_steps = 500
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=None)
        total_reward = 0

        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1

            if done:
                break

        print(f"Episode {ep:03d} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    
    env.close()
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
    print('Model Saved')
    return agent


if __name__ == "__main__":
    train()
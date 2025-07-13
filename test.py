import torch
import gymnasium as gym
import time
from agent import DQNAgent

def load_agent(path, state_dim, action_dim, device="cuda" if torch.cuda.is_available() else "cpu"):
    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.policy_net.load_state_dict(torch.load("dqn_cartpole.pth"))
    agent.policy_net.eval()
    return agent

    
def test(agent, episodes=50):
    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
           with torch.no_grad():
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.01)

        print(f"Test Episode {ep}: Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_agent("dqn_cartpole.pth", state_dim, action_dim, device)
    test(agent)

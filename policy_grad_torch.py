import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# Physical Constants
m_c = 1.0    # mass of the cart (kg)
m_p = 1.0    # mass of the pole (kg)
l = 1        # length of the pole (m)
g = 9.81     # gravity (m/s^2)
mu_c = 0.02  # cart friction coefficient
mu_p = 0.02  # pole friction coefficient

# Network Constants
layer_sizes = [4, 64, 64, 64, 1]

# Simulation/Trajectory Constants
dt = 0.02                # length of each time step in seconds
trajectory_steps = 500   # each trajectory contains this many time steps
trajectory_time = dt * trajectory_steps

# Training Constants
n_trajectories = 100     # number of trajectories to generate per epoch
n_epochs = 10001         # number of epochs to train for
max_x = 1.0              # initial positions are in (-max_x, max_x) meters
max_v = 0.5              # initial velocities are in (-max_v, max_v) meters/second
max_theta = np.pi        # initial thetas are in (-max_theta, max_theta) radians (don't change this)
max_omega = 0.5 * np.pi  # initial omegas are in (-max_omega, max_omega) radians/second

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trajectory_step(state, f, dt=0.1, g=9.81, m_c=1.0, m_p=0.1, l=0.5, mu_c=0.1, mu_p=0.05):
    """
    state: [batch_size, 4] -> (x, v, theta, omega)
    f: [batch_size] or [batch_size, 1] -> force
    returns: [batch_size, 4] -> next state
    """
    x, v, theta, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    A1 = m_c + m_p  # scalar
    B1 = 0.5 * m_p * l * cos_theta
    C1 = f - mu_c * v + 0.5 * m_p * l * omega**2 * sin_theta

    A2 = B1
    B2 = (1/3) * m_p * l**2  # scalar
    C2 = -mu_p * omega + 0.5 * m_p * l * g * sin_theta

    # Construct matrices for batched linear solve: M @ [a, alpha] = RHS
    M11 = torch.full_like(B1, A1)
    M12 = B1
    M21 = A2
    M22 = torch.full_like(B1, B2)
    
    M = torch.stack([
        torch.stack([M11, M12], dim=-1),
        torch.stack([M21, M22], dim=-1)
    ], dim=1)  # shape: [batch, 2, 2]

    RHS = torch.stack([C1, C2], dim=-1).unsqueeze(-1)  # [batch, 2, 1]

    sol = torch.linalg.solve(M, RHS).squeeze(-1)  # [batch, 2]
    a, alpha = sol[:, 0], sol[:, 1]

    # Integrate forward
    x_new = x + v * dt
    v_new = v + a * dt
    theta_new = theta + omega * dt
    omega_new = omega + alpha * dt

    return torch.stack([x_new, v_new, theta_new, omega_new], dim=1)


class PolicyNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(PolicyNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Tanh())
        # Last layer (no activation)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.network = nn.Sequential(*layers)

        # Log standard deviation as a learnable parameter
        self.log_std_dev = nn.Parameter(torch.tensor([4.0], dtype=torch.float32))

        # Xavier init
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        # Normalize theta to [-pi, pi]
        x = state.clone()
        x[:, 2] = (x[:, 2] + np.pi) % (2 * np.pi) - np.pi
        return self.network(x)

    def sample_force(self, state, noise):
        mean_force = self.forward(state)
        std_dev = torch.exp(self.log_std_dev)
        return mean_force.T + std_dev * noise
    

def get_initial_state(n_samples, max_x=max_x, max_v=max_v, max_theta=max_theta, max_omega=max_omega):
    scale = torch.tensor([max_x, max_v, max_theta, max_omega])
    noise = 2* torch.rand((n_samples, 4)) - 1
    return scale*noise


def get_trajectory(state, policy: PolicyNetwork, T, dt=0.1, g=9.81, m_c=1.0, m_p=0.1, l=0.5, mu_c=0.1, mu_p=0.05):
    """
    Roll out trajectory using a policy that maps state to force.

    state: [batch_size, 4] -> initial state
    policy: function(state) -> force (shape: [batch_size] or [batch_size, 1])
    T: number of steps
    returns: [T+1, batch_size, 4] -> trajectory of states
    """
    traj = [state]
    all_actions = []
    noise = torch.normal(0.0, 1.0, (state.shape[0], trajectory_steps))

    for t in range(T):
        f_t = policy.sample_force(state, noise[:, t]).squeeze(0)
        all_actions.append(f_t)
        state = trajectory_step(state, f_t, dt=dt, g=g, m_c=m_c, m_p=m_p, l=l, mu_c=mu_c, mu_p=mu_p)
        traj.append(state)

    return torch.stack(traj, dim=0), torch.stack(all_actions, dim=0)


def reward_fn(new_state):
    x, v, theta, omega = new_state.unbind(-1)

    # Normalize angle to [-pi, pi]
    normalized_theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

    theta_reward = -dt * normalized_theta**2
    x_reward = -dt * x**2

    return theta_reward + 0.001 * x_reward


def plot_trajectories(states, filename="trajectory.png", alpha=0.5):
    fig = plt.figure()
    axs = fig.subplots(nrows=4, ncols=1)

    axs[3].set_xlabel("time (s)")
    axs[0].set_ylabel("x (m)")
    axs[1].set_ylabel("v (m/s)")
    axs[2].set_ylabel("theta (rad)")
    axs[3].set_ylabel("omega (rad/s)")

    for state in states:
        x, v, theta, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        axs[0].plot(x.detach().numpy(), alpha=alpha)
        axs[1].plot(v.detach().numpy(), alpha=alpha)
        axs[2].plot(theta.detach().numpy(), alpha=alpha)
        axs[3].plot(omega.detach().numpy(), alpha=alpha)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def train(policy, n_epochs, optimizer, print_every=100, plot_every=1000):
    for epoch in range(n_epochs):
        # 1. Sample initial states
        init_states = get_initial_state(n_trajectories).to(device).float()

        # 2. Sample trajectories
        trajectories, actions = get_trajectory(init_states, policy, trajectory_steps)  # shape: [T+1, B, 4]
        # 3. Compute rewards
        rewards = reward_fn(trajectories[1:])  # skip initial state, shape: [T, B]
        returns = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])  # discounted sum of future rewards (γ=1)

        # 4. Compute log probs of actions
        states = trajectories[:-1].reshape(-1, 4)
        actions = actions.reshape(-1, 1)  # the actions actually taken
        means = policy(states)
        std_dev = torch.exp(policy.log_std_dev)

        log_probs = -0.5 * ((actions - means) / std_dev).pow(2) - torch.log(std_dev) - 0.5 * np.log(2 * np.pi)

        # 5. Policy loss (REINFORCE): maximize expected return = minimize -log_prob * return
        loss = -(log_probs.squeeze() * returns.flatten()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6. Logging
        if epoch % print_every == 0:
            avg_reward = rewards.mean().item()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.4f}")

        # 7. Optional: save plots
        if epoch % plot_every == 0:
            with torch.no_grad():
                test_states = get_initial_state(10).to(device).float()
                test_traj, test_actions = get_trajectory(test_states, policy, trajectory_steps)
                plot_trajectories(test_traj.cpu(), f"trajectory_{epoch}.png", alpha=0.7)


policy = PolicyNetwork(layer_sizes).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
train(policy, n_epochs, optimizer)

from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
n_trajectories = 500     # number of trajectories to generate per epoch
n_epochs = 10001         # number of epochs to train for
max_x = 1.0              # initial positions are in (-max_x, max_x) meters
max_v = 0.5              # initial velocities are in (-max_v, max_v) meters/second
max_theta = np.pi        # initial thetas are in (-max_theta, max_theta) radians (don't change this)
max_omega = 0.5 * np.pi  # initial omegas are in (-max_omega, max_omega) radians/second

data_dir = Path("data/policy_grad/")


def trajectory_step(state, f, dt=0.02, np=np):
    """
    given a state of the form [x, v, theta, omega, a force f, and a time dt, 
    return the new state of the system after applying the force on the cart for dt time.
    """
    x, v, theta, omega = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    A1 = m_c + m_p
    B1 = 0.5 * m_p * l * cos_theta
    C1 = f - mu_c * v + 0.5 * m_p * l * omega**2 * sin_theta

    A2 = B1
    B2 = (1/3) * m_p * l**2
    C2 = -mu_p * omega + 0.5 * m_p * l * g * sin_theta

    # Matrix form:
    M = np.array([[A1, B1],
                  [A2, B2]])
    RHS = np.array([C1, C2])

    # Solve for a and alpha
    a, alpha = np.linalg.solve(M, RHS)

    x_new = x + v * dt
    v_new = v + a * dt
    theta_new = theta + omega * dt
    omega_new = omega + alpha * dt

    return np.array([x_new, v_new, theta_new, omega_new])


def initialize_params(key, layer_sizes):
    n_layers = len(layer_sizes) - 1
    params = { "weights": [], "biases": [], "log_std_dev": jnp.array([4.0]) }
    keys = jax.random.split(key, n_layers)
    for layer in range(n_layers):
        in_size = layer_sizes[layer]
        out_size = layer_sizes[layer + 1]
        # use xavier initialization for weights: std dev of 1/sqrt(n) for n inputs
        params["weights"].append(
            1.0 / np.sqrt(in_size) * jax.random.normal(keys[layer], shape=(out_size, in_size)),
        )
        # initialize biases to zero
        params["biases"].append(
            jnp.zeros((out_size)),
        )
    return params


def write_params(params, filename):
    weights = { f"weights_{i}": params["weights"][i] for i in range(len(params["weights"])) }
    biases = { f"biases_{i}": params["biases"][i] for i in range(len(params["biases"])) }
    np.savez(filename, **weights, **biases, log_std_dev=params["log_std_dev"])


def read_params(filename):
    npz_file = np.load(filename)
    # n_files = 2 * n + 1 where n is the number of layers
    n_files = len(npz_file.files)
    n_layers = (n_files - 1) // 2
    return {
        "weights": [npz_file[f"weights_{i}"] for i in range(n_layers)],
        "biases": [npz_file[f"biases_{i}"] for i in range(n_layers)],
        "log_std_dev": npz_file["log_std_dev"],
    }


def compute_mean_force(state, params):
    """
    Defines our stochastic policy.

    It takes a state (x, v, theta, omega) in (m, m/s, rad, rad/s), and a PRNG key, and returns the
    mean of the force distribution (array of shape (1,)) in Newtons.
    """
    n_layers = len(params["weights"])
    # Wrap theta to the interval [-pi, pi].
    theta = (state[2] + np.pi) % 2 - np.pi
    x = jnp.array([state[0], state[1], theta, state[3]])
    for layer in range(n_layers - 1):
        x = params["weights"][layer] @ x + params["biases"][layer]
        x = jnp.tanh(x)
    # The last layer doesn't have an activation function.
    return params["weights"][-1] @ x + params["biases"][-1]


def sample_force(state, noise, params):
    """
    Generates a sample (array of shape (1,)) from our stochastic policy, for the given state and
    unit normal sample.
    """
    return compute_mean_force(state, params) + jnp.exp(params["log_std_dev"]) * noise


def generate_initial_conditions(key, n_samples, max_x, max_v, max_omega):
    """
    Generates uniformly distributed samples (x, v, theta, omega) in phase space.

    The x samples are in the range [-max_x, max_x]. The v and omega samples are in the range
    [-max_v, max_v] and [-max_omega, max_omega]. Theta samples are always in the range [-pi, pi].
    """
    scale = np.array([max_x, max_v, max_theta, max_omega])
    noise = jax.random.uniform(key, shape=(n_samples, 4), minval=-1.0, maxval=1.0)
    return scale * noise


def sample_trajectory(key, initial_state, params, length, dt):
    noise = jax.random.normal(key, shape=(length,))

    # Each row stores (x, v, theta, omega, force, reward). The force and reward in the last row
    # won't be filled in, since we don't choose an action for that state.
    trajectories = jnp.zeros((length + 1, 6))
    trajectories = trajectories.at[0, 0:4].set(initial_state)

    def body_fn(i, trajectories):
        state = trajectories[i, 0:4]
        force = sample_force(state, noise[i], params)[0]
        new_state = trajectory_step(state, force, dt, np=jnp)

        # The reward is the integral of minus the square of the angle from vertical (in radians).
        theta_reward = -dt * jnp.square((new_state[2] + np.pi) % (2 * np.pi) - np.pi)
        # We also add a penalty if the cart moves too far to either side.
        x_reward = -dt * jnp.square(new_state[0])
        reward = theta_reward + 0.001 * x_reward

        trajectories = trajectories.at[i, 4].set(force)
        trajectories = trajectories.at[i, 5].set(reward)
        trajectories = trajectories.at[i + 1, 0:4].set(new_state)
        return trajectories

    return jax.lax.fori_loop(0, length, body_fn, trajectories)


def sample_trajectories(key, initial_states, params, n_trajectories, length, dt):
    keys = jax.random.split(key, n_trajectories)
    sample_batch = jax.vmap(sample_trajectory, in_axes=(0, 0, None, None, None))
    return sample_batch(keys, initial_states, params, length, dt)


def estimate_policy_gradient(trajectories, params):
    def compute_force_log_prob(state, force, params):
        mean_force = compute_mean_force(state, params)
        prob = jax.scipy.stats.norm.pdf(force, loc=mean_force, scale=jnp.exp(params["log_std_dev"]))
        return jnp.log(prob)[0]

    def compute_force_log_probs_for_trajectory(trajectory, params):
        flp = lambda state: compute_force_log_prob(state[0:4], state[4], params)
        return jax.vmap(flp)(trajectory[0:-1])

    def sum_log_probs_and_rewards_for_trajectory(trajectory, params):
        rewards = trajectory[0:-1, 5]
        rewards_to_go = jnp.flip(jnp.cumsum(jnp.flip(rewards)))
        lps = compute_force_log_probs_for_trajectory(trajectory, params)
        return jnp.dot(lps, rewards_to_go)

    def average_log_probs_and_rewards_for_batch(trajectories, params):
        sum_log_probs_and_rewards_for_batch = jax.vmap(
            sum_log_probs_and_rewards_for_trajectory,
            in_axes=(0, None)
        )
        batch_log_prob_sums = sum_log_probs_and_rewards_for_batch(trajectories, params)
        return jnp.mean(batch_log_prob_sums, axis=0)

    return jax.grad(average_log_probs_and_rewards_for_batch, argnums=1)(trajectories, params)


def plot_training_stats(stats, filename):
    steps = np.arange(len(stats["force_log_std_devs"]))

    fig = plt.figure(figsize=(12, 12))
    axs = fig.subplots(nrows=3, ncols=1)
    axs[1].set_xlabel("training step")

    reward_quantiles = stats["reward_quantiles"]
    n_quantiles = reward_quantiles.shape[1] - 1
    axs[0].set_ylabel("reward")
    axs[0].plot(steps, reward_quantiles[:, n_quantiles // 2], color="steelblue")
    for i in range(n_quantiles // 2):
        axs[0].fill_between(
            steps,
            reward_quantiles[:, i],
            reward_quantiles[:, n_quantiles - i],
            alpha=0.1,
            color="steelblue",
        )

    final_theta_quantiles = stats["final_theta_quantiles"]
    n_quantiles = final_theta_quantiles.shape[1] - 1
    axs[1].set_ylabel("final theta")
    axs[1].plot(steps, final_theta_quantiles[:, n_quantiles // 2], color="darkorchid")
    for i in range(n_quantiles // 2):
        axs[1].fill_between(
            steps,
            final_theta_quantiles[:, i],
            final_theta_quantiles[:, n_quantiles - i],
            alpha=0.1,
            color="darkorchid",
        )

    axs[2].set_ylabel("force log std dev")
    axs[2].plot(steps, stats["force_log_std_devs"])

    fig.tight_layout()
    fig.savefig(data_dir / filename)
    plt.close(fig)


def plot_trajectories(states, filename = "trajectory.png", alpha = 0.5):
    for idx in range(1, len(states)):
        assert np.all(states[idx][:, 0] == states[0][:, 0])

    fig = plt.figure()
    axs = fig.subplots(nrows=4, ncols=1)

    axs[3].set_xlabel("time (s)")
    axs[0].set_ylabel("x (m)")
    axs[1].set_ylabel("v (m/s)")
    axs[2].set_ylabel("theta (rad)")
    axs[3].set_ylabel("omega (rad/s)")

    for state_array in states:
        axs[0].plot(state_array[:, 0], state_array[:, 1], alpha=alpha)
        axs[1].plot(state_array[:, 0], state_array[:, 2], alpha=alpha)
        axs[2].plot(state_array[:, 0], state_array[:, 3], alpha=alpha)
        axs[3].plot(state_array[:, 0], state_array[:, 4], alpha=alpha)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def train(params: optax.Params, optimizer: optax.GradientTransformation, key) -> optax.Params:
    opt_state = optimizer.init(params)
    n_quantiles_groups = 4
    all_reward_quantiles = []
    all_final_theta_quantiles = []
    all_force_log_std_devs = []

    @jax.jit
    def step(params, opt_state, key):
        key, key0, key1 = jax.random.split(key, 3)
        initial_states = generate_initial_conditions(key0, n_trajectories, max_x, max_v, max_omega)
        trajectories = sample_trajectories(
            key1,
            initial_states,
            params,
            n_trajectories,
            trajectory_steps,
            dt,
        )
        policy_grad = estimate_policy_gradient(trajectories, params)
        minus_policy_grad = jax.tree.map(lambda x: -x, policy_grad)
        updates, opt_state = optimizer.update(minus_policy_grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, trajectories

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, trajectories = step(params, opt_state, subkey)

        # Print info about this epoch.
        # The raw batch rewards are in units of radians^2 * seconds (and are all negative).
        # Our mean and RMSE reward metrics are in units of radians.
        # If the pendulum were fixed in place for a full trajectory, that trajectory's normalized
        # reward would just be its angle from vertical in radians.
        batch_rewards = jnp.sum(trajectories[:, :, 5], axis=1) / trajectory_time
        quantiles = jnp.linspace(0.0, 1.0, n_quantiles_groups + 1)
        reward_quantiles = jnp.quantile(batch_rewards, quantiles)
        final_thetas = jnp.mod(trajectories[:, -1, 2] + np.pi, 2 * np.pi) - np.pi
        final_theta_quantiles = jnp.quantile(final_thetas, quantiles)
        all_reward_quantiles.append(reward_quantiles)
        all_final_theta_quantiles.append(final_theta_quantiles)
        log_std_dev = params['log_std_dev']
        all_force_log_std_devs.append(log_std_dev)
        print(
            f"epoch {epoch}, " +
            f"reward quantiles: {reward_quantiles}, " +
            f"final theta quantiles: {final_theta_quantiles}, " +
            f"force std dev: {jnp.exp(log_std_dev)}"
        )

        # Save trajectories and params.
        if epoch % 200 == 0:
            write_params(params, data_dir / "params" / f"epoch_{epoch:05}_params.npz")
            jnp.save(data_dir / "trajectories" / f"epoch_{epoch:05}_trajectories.npy", trajectories, allow_pickle=False)

            stats = {
                "reward_quantiles": np.array(all_reward_quantiles),
                "final_theta_quantiles": np.array(all_final_theta_quantiles),
                "force_log_std_devs": np.array(all_force_log_std_devs),
            }
            plot_training_stats(stats, Path("training") / f"epoch_{epoch:05}_stats.png")
            np.savez(data_dir / "training" / "stats.npz", **stats)

            times = dt * np.arange(trajectory_steps + 1)
            times = times.reshape((-1, 1))
            vis_trajectories = [np.concatenate([times, trajectories[i]], axis=1) for i in range(50)]
            for traj in vis_trajectories:
                traj[:, 3] = np.mod(traj[:, 3] + np.pi, 2 * np.pi) - np.pi
            plot_trajectories(
                vis_trajectories,
                data_dir / "training" / f"epoch_{epoch:05}_trajectories.png",
            )

    stats = {
        "reward_quantiles": np.array(all_reward_quantiles),
        "final_theta_quantiles": np.array(all_final_theta_quantiles),
        "force_log_std_devs": np.array(all_force_log_std_devs),
    }
    return params, stats


if __name__ == "__main__":
    # key = jax.random.PRNGKey(123456)
    seed = np.random.randint(-9223372036854775808, 9223372036854775807)
    print(f"PRNG seed: {seed}")
    key = jax.random.PRNGKey(seed)

    (data_dir / "params").mkdir(parents=True, exist_ok=True)
    (data_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (data_dir / "training").mkdir(parents=True, exist_ok=True)

    key, subkey = jax.random.split(key)
    params = initialize_params(subkey, layer_sizes)

    key, subkey = jax.random.split(key)
    optimizer = optax.adam(learning_rate=1e-3)
    params = train(params, optimizer, subkey)
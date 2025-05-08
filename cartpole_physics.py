import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp

m_c = 1.0
m_p = 1.0
l = 1
g = 9.81
mu_c = 0.02
mu_p = 0.02

def trajectory_step(state, f, dt=0.02, np=np):
    x, v, theta, omega = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    A1 = m_c + m_p
    B1 = 0.5 * m_p * l * cos_theta
    C1 = f - mu_c * v + 0.5 * m_p * l * omega**2 * sin_theta

    A2 = B1
    B2 = (1/3) * m_p * l**2
    C2 = -mu_p * omega + 0.5 * m_p * l * g * sin_theta

    M = np.array([[A1, B1],
                  [A2, B2]])
    RHS = np.array([C1, C2])

    a, alpha = np.linalg.solve(M, RHS)

    x_new = x + v * dt
    v_new = v + a * dt
    theta_new = theta + omega * dt
    omega_new = omega + alpha * dt

    return np.array([x_new, v_new, theta_new, omega_new])

def animate_cartpole(trajectory, l=0.5, interval=20):
    plt.switch_backend('Agg')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-1.5, 2.5)

    ax.plot([-7, 7], [0, 0], 'k', lw=2)

    cart_width, cart_height = 0.8, 0.6
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
    pole_line, = ax.plot([], [], lw=6, c='orange')
    ax.add_patch(cart_patch)

    def init():
        cart_patch.set_xy((-cart_width / 2, 0))
        pole_line.set_data([], [])
        return cart_patch, pole_line

    def update(frame):
        x, _, theta, _ = trajectory[frame]

        cart_patch.set_xy((x - cart_width / 2, 0))

        pole_x = [x, x + (l+0.5) * np.sin(theta)]
        pole_y = [cart_height / 2, cart_height / 2 + (l+0.5) * np.cos(theta)]
        pole_line.set_data(pole_x, pole_y)

        return cart_patch, pole_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectory),
        init_func=init, blit=True, interval=interval, repeat=True
    )

    plt.grid(True)
    plt.tight_layout()
    
    print("Saving animation to 'cartpole_animation.mp4'...")
    ani.save('cartpole_animation.mp4', 
             writer='ffmpeg', 
             fps=30,
             dpi=100,
             progress_callback=lambda i, n: print(f"\rProgress: {i}/{n} frames ({i/n*100:.1f}%)", end=""))
    print("\nAnimation saved successfully!")
    plt.close()

def read_params(filename):
    npz_file = np.load(filename)
    n_files = len(npz_file.files)
    n_layers = (n_files - 1) // 2
    return {
        "weights": [npz_file[f"weights_{i}"] for i in range(n_layers)],
        "biases": [npz_file[f"biases_{i}"] for i in range(n_layers)],
        "log_std_dev": npz_file["log_std_dev"],
    }

def compute_mean_force(state, params):
    n_layers = len(params["weights"])
    theta = (state[2] + np.pi) % 2 - np.pi
    x = jnp.array([state[0], state[1], theta, state[3]])
    for layer in range(n_layers - 1):
        x = params["weights"][layer] @ x + params["biases"][layer]
        x = jnp.tanh(x)
    return params["weights"][-1] @ x + params["biases"][-1]

params = read_params('./data/policy_grad/params/epoch_08000_params.npz')

state = np.array([-1.54, 0.0, np.pi, 0.0])
trajectory = [state]

for i in range(500):
    f = compute_mean_force(state, params).item()
    state = trajectory_step(state, f)
    trajectory.append(state)

animate_cartpole(trajectory)

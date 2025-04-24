import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
m_c = 1.0    # mass of the cart (kg)
m_p = 1.0    # mass of the pole (kg)
l = 1        # length of the pole (m)
g = 9.81     # gravity (m/s^2)
mu_c = 0.02  # cart friction coefficient
mu_p = 0.02  # pole friction coefficient

def cartpole_dynamics(state, f, dt=0.02):
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



def animate_cartpole(trajectory, l=0.5, interval=20):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 1.5)

    # Draw track
    ax.plot([-2.5, 2.5], [0, 0], 'k', lw=2)

    # Elements to animate
    cart_width, cart_height = 0.3, 0.2
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
    pole_line, = ax.plot([], [], lw=4, c='orange')
    ax.add_patch(cart_patch)

    def init():
        cart_patch.set_xy((-cart_width / 2, 0))
        pole_line.set_data([], [])
        return cart_patch, pole_line

    def update(frame):
        x, _, theta, _ = trajectory[frame]

        # Update cart
        cart_patch.set_xy((x - cart_width / 2, 0))

        # Update pole
        pole_x = [x, x + l * np.sin(theta)]
        pole_y = [cart_height / 2, cart_height / 2 + l * np.cos(theta)]
        pole_line.set_data(pole_x, pole_y)

        return cart_patch, pole_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectory),
        init_func=init, blit=True, interval=interval, repeat=False
    )

    plt.title("Cart-Pole Animation")
    plt.grid(True)
    plt.show()


state = np.array([-2.0, 0.0, np.pi, 0.0])
trajectory = [state]
for i in range(2000):
    f = 2*np.sin(i/10)
    state = cartpole_dynamics(state, f=f)
    trajectory.append(state)

# Animate it
animate_cartpole(trajectory)

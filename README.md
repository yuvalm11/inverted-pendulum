# Solving the cartpole problem with Reinforcement Learning

I used the cartpole problem as a convenient example to learn about Reinforcement Learning algorithms, specifically Vanilla Policy Gradient. In this document I will share the steps I took, starting with modeling the dynamics of the system, then implementing the algorithm, and finally some analysis of the results.

## Introduction

The cartpole problem is a classic RL control problem where an agent is trained to control a horizontally moving cart with a pole or a pendulum attached to it, such that the pole is balanced in its upright position.

![](https://images.ctfassets.net/xjan103pcp94/4hLHnMXJN2EwwAXq2yYx9v/41b16121290d6c46b6b85492a572a4cf/cartPoleRemade.png)

## Modelling The System

We'll use the Lagrangian approach to derive the dynamic equations of the system. Specifically, we seek to find equations that describe the motion of the cart and the pole after applying a certain force on the cart.

The Lagrangian function of the system is given by:

$$L = T - V$$

where $T$ is the kinetic energy of the system and $V$ is the potential energy.

From the Lagrangian, we can derive the Euler-Lagrange equations:

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q_i}} \right) - \frac{\partial L}{\partial{q_i}} = Q_i$$

where $q_i$ are the generalized coordinates of the system, $\dot{q_i}$ are the generalized velocities, and $Q_i$ are the generalized forces.

In the case of the cartpole, the generalized coordinates will be the position of the cart $x$ and the angle of the pole $\theta$ (counter clockwise from the upright position). The generalized velocities will be the velocity of the cart $\dot{x}=v$ and the angular velocity of the pole $\dot{\theta}=\omega$. The generalized forces will be the force applied to the cart $f$ and the friction force.

Using this setup, we now have a basis of the equations that govern the motion of the cartpole. We can now go through some of the math to derive the exact equations that, hopefully, will let us find how the cart and pole move in time after applying a force -- that is, find $\ddot{x} = a$ and $\ddot{\theta} = \alpha$, the accelerations of the cart and the pole. 

Let's define:

- $m_c$ - the mass of the cart
- $m_p$ - the mass of the pole
- $l$ - the length of the pole
- $g$ - the acceleration due to gravity
- $\mu_c$ - friction coefficient of the cart
- $\mu_p$ - friction coefficient of the pole


### Kinetic Energy:

The kinetic energy of the cart is simply given by the following equation: $$T_\text{cart} = \frac{1}{2}m_cv^2$$

To find the kinetic energy of the pole, we'll view it as a collection of small point-like objects along the pole and integrate over the pole. 

Therefore we  define:
- $dm_p$ - the mass of each small point-like object
- $r$ - the distance from the point to the cart
- $dr$ - the length of the point-like object
- $x_s, y_x$ - the x, y coordinates of the point

First, since the mass is uniformly distributed:
$$
\frac{dm_p}{m_p} = \frac{dr}{l} \Rightarrow dm_p = \frac{m_p}{l}dr
$$

Now, we can express $x_s, y_x$ and $\dot{x_s}, \dot{y_x}$:

$$
\begin{aligned}
&x_s = x+r\sin\theta &\ \ \ & y_s = r\cos\theta \\
&\dot{x_s} = v+\omega r\cos\theta && \dot{y_s} = -\omega r\sin\theta
\end{aligned}
$$

Therefore, the squared speed of each object is:
$$
v_s^2 = \dot{x_s}^2 + \dot{y_s}^2 = v^2 + 2v\omega r \cos\theta + \omega^2r^2
$$

and the kinetic energy of each object is:
$$
dT_\text{pole} = \frac{1}{2}dm_p\cdot v_s^2 = \frac{m_p}{2l}(v^2 + 2v\omega r \cos\theta + \omega^2r^2)dr
$$

We can now integrate over the pole to get the kinetic energy of the pole:

$$
\begin{aligned}
\int_\text{pole} dT_\text{pole} &= \int_0^l \frac{m_p}{2l}(v^2 + 2v\omega r \cos\theta + \omega^2r^2)dr\\
T_\text{pole} &= \frac{m_p}{2l} \cdot \left( v^2 + v\omega l^2\cos\theta + \frac{\omega^2l^3}{3}\right)\\
T_\text{pole} &= \frac{1}{2}m_pv^2 + \frac{1}{2}m_pv\omega l\cos\theta + \frac{1}{6}m_p\omega^2l^2
\end{aligned}
$$


And the total kinetic energy of the system:
$$
\begin{aligned}
T 
&= T_\text{cart} + T_\text{pole}\\
&= \frac{1}{2}m_cv^2 + \frac{1}{2}m_pv^2 + \frac{1}{2}m_pv\omega l\cos\theta + \frac{1}{6}m_p\omega^2l^2\\
&= \frac{1}{2}(m_c+m_p)v^2 + \frac{1}{2}m_p\omega l\left(v\cos\theta+\frac{1}{3}\omega l\right)
\end{aligned}
$$

### Potential Energy:

The only potential energy in the system is the gravitational potential energy of the pole. Similarly, we will integrate over the pole to find it:

$$
\begin{aligned}
dV &= dm_p \cdot g \cdot y_s\\
dV &= \frac{m_pg\cos\theta}{l}\cdot rdr\\
\int_\text{pole} dV &= \frac{m_pg\cos\theta}{l}\int_0^l rdr
V = \frac{1}{2}m_pgl\cos\theta
\end{aligned}
$$

### The Lagrangian Function:

$$
L = T - V = 
\frac{1}{2}(m_c+m_p)v^2 + \frac{1}{2}m_p\omega l\left(v\cos\theta+\frac{1}{3}\omega l\right)
-
\frac{1}{2}m_pgl\cos\theta
$$

## Find The Acceleration

Recall from the beginning the Euler-Lagrange equations:

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q_i}} \right) - \frac{\partial L}{\partial{q_i}} = Q_i$$

Since $q_i=x$, we can see that $\frac{\partial L}{\partial{q_i}} =\frac{\partial L}{\partial{x}} =0$. We can now find $\frac{\partial L}{\partial \dot{q_i}}$:

$$
\frac{\partial L}{\partial v} = (m_c+m_p)\cdot v + \frac{1}{2}m_p\omega l\cos\theta
$$
and
$$
\frac{d}{dt} \left( \frac{\partial L}{\partial v} \right) = (m_c + m_p) \cdot a + \frac{1}{2}m_pl(\alpha\cos\theta - \omega^2\sin\theta)
$$

By approximating the damping force on the cart we get: 

$$
\begin{aligned}
Q_x = f - \mu_cv
&= \frac{d}{dt} \left( \frac{\partial L}{\partial v} \right) - \cancel{\frac{\partial L}{\partial{x}}}\\

f - \mu_cv &= (m_c + m_p) \cdot a + \frac{1}{2}m_pl(\alpha\cos\theta - \omega^2\sin\theta)
\end{aligned}
$$

Similarly on the pole we get:
$$
\frac{\partial L}{\partial{q_i}} = \frac{\partial L}{\partial{\theta}} = \frac{1}{2}m_pl\sin\theta(g-v\omega)
$$
and
$$
\frac{\partial L}{\partial \dot{q_i}} = \frac{\partial L}{\partial \omega} = \frac{1}{2}m_plv\cos\theta + \frac{1}{3}m_pl^2\omega
$$
Therefore: 
$$
\frac{d}{dt} \left( \frac{\partial L}{\partial \omega} \right) = \frac{1}{2}m_pl(a\cos\theta - v\omega\sin\theta) + \frac{1}{3}m_pl^2\alpha
$$
Finally we get:
$$
\begin{aligned}
-\mu_p\omega &= \frac{1}{2}m_pl(a\cos\theta - v\omega\sin\theta) + \frac{1}{3}m_pl^2\alpha - \frac{1}{2}m_pl\sin\theta(g-v\omega)\\
&=\frac{1}{2}m_pl(a\cos\theta - g\sin\theta ) + \frac{1}{3}m_pl^2\alpha
\end{aligned}
$$

We can now do some algebra to get the equations of $\ddot{x}$ and $\ddot{\theta}$:

Given the equations:

$$
\begin{cases}
f - \mu_cv = (m_c + m_p) \cdot a + \frac{1}{2}m_pl(\alpha\cos\theta - \omega^2\sin\theta)
\\\\
-\mu_p\omega = \frac{1}{2}m_pl(a\cos\theta - g\sin\theta ) + \frac{1}{3}m_pl^2\alpha
\end{cases}
$$

We can rewrite them as follows:

$$
\begin{bmatrix}
m_c+m_p & \frac{1}{2}m_pl\cos\theta  \\\\
\frac{1}{2}m_pl\cos\theta & \frac{1}{3}m_pl^2
\end{bmatrix}
\begin{bmatrix}
a \\\\
\alpha
\end{bmatrix} =
\begin{bmatrix}
f-\mu_cv+\frac{1}{2}m_pl\omega^2\sin\theta \\\\
-\mu_p\omega+\frac{1}{2}m_plg\sin\theta
\end{bmatrix}
$$

We define:

$$
\begin{aligned}
&A_1 = m_c+m_p  && B_1 = \frac{1}{2}m_pl\cos\theta && C_1 = f-\mu_cv+\frac{1}{2}m_pl\omega^2\sin\theta -\mu_p\omega+\frac{1}{2}m_plg\sin\theta\\
&A_2 = \frac{1}{2}m_pl\cos\theta  && B_2 = \frac{1}{3}m_pl^2  && C_2 = 
\end{aligned}
$$

So we get the following system of equations:

$$
\begin{bmatrix}
A_1 & B_1\\\\
A_2 & B_2
\end{bmatrix}
\begin{bmatrix}
a \\\\
\alpha
\end{bmatrix} =
\begin{bmatrix}
C_1 \\\\
C_2
\end{bmatrix}
$$

Using Cramer's rule we get:

$$
\boxed{
a = \frac{
    \begin{vmatrix}
        C_1 & B_1 \\
        C_2 & B_2
    \end{vmatrix}
}{
    \begin{vmatrix}
        A_1 & B_1 \\
        A_2 & B_2
    \end{vmatrix}
} = \frac{
    C_1B_2 - B_1C_2
}{
    A_1B_2 - B_1A_2
}}
\ \ \ \ \
\boxed{
\alpha = \frac{
    \begin{vmatrix}
        A_1 & C_1 \\
        A_2 & C_2
    \end{vmatrix}
}{
    \begin{vmatrix}
        A_1 & B_1 \\
        A_2 & B_2
    \end{vmatrix}
} = \frac{
    A_1C_2 - C_1A_2
}{
    A_1B_2 - B_1A_2
}
}
$$


## Reinforcement Learning - Policy Gradient

We will use the Vanilla Policy Gradient algorithm to train our agent. We aim to learn an effective way to control the cart (i.e. apply the correct horizontal force on it to keep the pole up). More specifically, we want to learn a **stochastic policy** $\pi_\theta(a|s)$, the probability distribution to take a certain action $a$ given a state $s$ of the system - in our case, we want to find a good probability density function over the force we apply on the cart. 

Intuitively, we do so by letting the policy act on our environment and by sampling many instances of our policy in action we can evaluate the contribution of certain actions to our overall goal. Then, we iteratively change our policy such that more effective actions will be more probable in the future.

Mathematically speaking, the update rule for our policy will be: 
$$
\theta \leftarrow \theta + \alpha \cdot \nabla_\theta\log\pi_\theta(a_t|s_t)*R_t
$$

Where $\theta$ is the parameters of our model, $R_t$ is the cumulative reward or the return from time $t$, and $\alpha$ is the learning rate.

Our reward function is a weighted average of the squared distance from $\theta = 0$ and $x=0$, where the $x$ distance is weighted significantly lower than the $\theta$ distance. That way the agent learns to first balance the pole and then center the cart. 

In [Erik Starnd's](https://fab.cba.mit.edu/classes/864.23/people/Erik/policy-gradients/), He's going through the theory of the algorithm in more detail. I recommend reading through it as well as his implementation of the algorithm.


### Results:

After ~10000 epochs the agent successfully learned how to balance the pendulum and center it as can be seen in the following graphs:

![](/trajectories.png)
Each line in the above plot shows a trajectory of 10 seconds after a random initialization of the system.
We can see how the $\theta$ values quickly gets centered around 0 as the pole is balanced. By looking at the $x$ values we see how the cart is being centered right after balancing the pole.


The following graph shows the rewards, the final $\theta$ values, and the log standard deviation of the forces as the model trains. 
![](/stats.png)


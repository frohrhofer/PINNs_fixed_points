import numpy as np
import matplotlib.pyplot as plt


def learning_curves(log):
    
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Plot loss curves
    epochs = np.arange(0, log['N_epochs'], log['freq_log'])
    for loss in ['loss_IC', 'loss_P']:
        ax.plot(epochs, log[loss], lw=1, label=loss)

    # Axis appearance
    ax.set_title('Learning Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(ls='--')
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    

def pendulum_dynamics(PINN):
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))

    ############################
    # Prediciton plot
    ############################

    # stable/unstable fixed points
    line_props = {'ls': '--', 'lw': 0.5}
    axes[0].axhline(0, c='green', **line_props)
    axes[0].axhline(np.pi, c='red', **line_props)
    axes[0].axhline(-np.pi, c='red', **line_props)

    # Reference data and PINN prediction
    t_line, theta_true, omega_true = PINN.data.reference()
    theta_pred = PINN(t_line)
    omega_pred = PINN.omega(t_line)

    # make plot
    axes[0].plot(t_line, theta_true, c='blue', lw=1, label='Reference')
    axes[0].plot(t_line, theta_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[0].set_xlabel(r'$t$')
    axes[0].set_ylabel(r'$\theta$')
    axes[0].legend(frameon=False, loc=1, ncol=2, fontsize=8)
    axes[0].set_ylim([-3.5, 3.5])

    #################
    # Quiver plot (Phase Space)
    #################

    # Background arrows
    xscale, yscale, n_arrows = 1.2, 2, 10
    theta = np.linspace(-xscale*np.pi, xscale*np.pi, n_arrows)
    omega = np.linspace(-yscale*np.pi, yscale*np.pi, n_arrows)
    XX, YY = np.meshgrid(theta, omega)
    Y = np.vstack([XX.flatten(), YY.flatten()])
    t = np.zeros(len(Y))
    [dtheta, domega] = PINN.data.diff_equations(t, Y)
    theta, omega = XX.flatten(), YY.flatten()
    axes[1].quiver(theta, omega, dtheta, domega, color='0.5')
    # Fixed Points
    axes[1].scatter(np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(-np.pi, 0, edgecolors='r', facecolors='none')
    axes[1].scatter(0, 0, edgecolors='g', facecolors='none')
    # Axis lines
    axes[1].axhline(0, lw=1, ls='--', c='black')
    axes[1].axvline(0, lw=1, ls='--', c='black')

    # Plot trajectories
    axes[1].plot(theta_true, omega_true, c='blue', lw=1, label='Reference')
    axes[1].plot(theta_pred, omega_pred, c='red', lw=1, ls='--', label='Prediction')

    # Axis appearance
    axes[1].set_xlabel(r'$\theta$')
    axes[1].set_ylabel(r'$\omega$')
    axes[1].set_xticks([-np.pi, 0, np.pi])
    axes[1].set_xticklabels([r'$\pi$', 0, r'$\pi$'])                

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt


def learning_curves(log):
    
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Plot loss curves
    epochs = np.arange(0, log['N_epochs'], log['freq_log'])
    for loss in ['loss_IC', 'loss_BC', 'loss_AC']:
        ax.plot(epochs, log[loss], lw=1, label=loss)

    # Axis appearance
    ax.set_title('Learning Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(ls='--')
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    
    
def allen_cahn_mesh(PINN):    
    
    # get reference data and make prediction
    X_ref, u_ref = PINN.data.reference_mesh()
    u_pred = PINN(X_ref)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True,
                             figsize=(10, 2))

    # properties for tricontourf plot
    props = {'cmap': 'jet', 'levels': 50, 'vmin': -1, 'vmax': 1}
    # Reference
    axes[0].set_title('Reference')
    contf = axes[0].tricontourf(X_ref[:,1], X_ref[:,0], u_ref[:,0], **props)
    fig.colorbar(contf, ax=axes[0])

    # Prediction
    axes[1].set_title('Prediction')
    contf = axes[1].tricontourf(X_ref[:,1], X_ref[:,0], u_pred[:,0], **props)
    fig.colorbar(contf, ax=axes[1])

    # Axis appearance
    axes[0].set_yticks([-1, -0.5, 0, 0.5, 1.0])
    axes[0].set_ylabel(r'$x$')
    for ax in axes:
        ax.set_xlabel(r't')

    plt.tight_layout()
    plt.show()
    
    
def allen_cahn_xcut(PINN, time_steps=[0, 0.5, 1.0]):

    fig, axes = plt.subplots(1, len(time_steps), 
                             sharey=True,
                             figsize=(2*len(time_steps), 2))

    # fill axes with plots for each time step
    for ax, time in zip(axes, time_steps):
        # data along x
        X, u_true = PINN.data.reference_xcut(time)
        u_pred = PINN(X)
        # make plot
        ax.set_title(fr'$t={time}$')
        ax.plot(X[:,0], u_true, c='blue', lw=1, label='Reference')
        ax.plot(X[:,0], u_pred, c='red', lw=1, ls='--', label='Prediction')
        ax.set_xlabel(r'$x$')

    # Axis appearance
    axes[0].set_ylabel(r'$u(x)$')
    axes[0].legend(loc=2, fontsize=8)

    plt.tight_layout()
    plt.show()

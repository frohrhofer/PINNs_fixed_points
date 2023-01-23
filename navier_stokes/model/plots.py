import numpy as np
import matplotlib.pyplot as plt


def learning_curves(log):
    
    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Plot loss curves
    epochs = np.arange(0, log['N_epochs'], log['freq_log'])
    for loss in ['loss_IC', 'loss_BC', 'loss_NS']:
        ax.plot(epochs, log[loss], lw=1, label=loss)

    # Axis appearance
    ax.set_title('Learning Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(ls='--')
    ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()
    
    
def flow_field(PINN, time=6):
    
    # load reference data
    CFD_data = PINN.data.load_csv()
    # Extract data at selected time t
    data_t = CFD_data[CFD_data['t'] == time]
    X, U_true = PINN.data.features_and_labels(data_t)
    # Make prediction
    U_pred = PINN(X)

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True,
                             figsize=(8, 6))

    # Reference solution (CFD data)
    for i, (ax, label) in enumerate(zip(axes[:,0], ['u', 'v', 'p'])):
        ax.set_title(f"Reference ${label}(x,y)$")
        contf = ax.tricontourf(X[:,0], X[:,1], U_true[:,i], cmap='jet', levels=30)
        fig.colorbar(contf, ax=ax)

    # PINN prediction
    for i, (ax, label) in enumerate(zip(axes[:,1], ['u', 'v', 'p'])):
        ax.set_title(f"Predicted ${label}(x,y)$")
        contf = ax.tricontourf(X[:,0], X[:,1], U_pred[:,i], cmap='jet', levels=30)
        fig.colorbar(contf, ax=ax)

    # Axis appearance
    for ax in axes.flatten():
        circle = plt.Circle((0, 0), 0.45, facecolor='white', edgecolor='black', lw=0.5)
        ax.add_patch(circle)

        ax.set_xlim([-5, 10])
        ax.set_ylim([-5, 5]) 

    for ax in axes[-1,:]:
        ax.set_xlabel(r'y')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'x')

    plt.tight_layout()
    plt.show()
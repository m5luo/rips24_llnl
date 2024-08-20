import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

nstencils = 4

# Neural Network I/O
input_size  = 4                     # Input layer neurons
hidden_size = 50                    # Hidden layer neurons
output_size = 3                     # Output layer neurons
phi_size = input_size - 1           # Input stencil dimension
psi_size = output_size              # Output stencil dimension

# Vector Properties
max_m = 4                           # Max coarsening factor 
vecsize = 3 + 2*2*max_m             # Vector size to use on Loss Functions
veccntr = divmod(vecsize,2)[0]      # Index to vector center

# Network Hyper-Parameters
eps = 1e-2                          # Epsilon factor
epochs = 150                        # Number of Epochs NN is trained
batch_size = 4                       
learning_rate = 1e-4                # Learning Rate (LR)

# Optimizer Parameters
momentum = 0.5                      # Parameter for SGD optimizer
learn_rate_drop_factor = 0.5        # Decay
learn_rate_drop_period = 10         # Number of past epochs to decrease LR
eps_drop_period = 5                 # Number of past epochs to decrease Epsilon

testing_dim = torch.randint(25, 50, size = [5])

plot_landscape=False
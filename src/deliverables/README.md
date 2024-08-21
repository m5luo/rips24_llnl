# Deliverables directions 🧭
This folder contains four files: `train_psi.ipynb`, `constants.py`, `loss_functions.py` and `helper_functions.py`

Running `train_psi.ipynb` will create a training dataset, train a neural network model with parameters defined in `constants.py`, then run PyMGRIT with the trained model. The parameters used, the model and any generated graphics will be saved in a folder.

To modify neural network parameters, see `constants.py`.

To change the loss function used by the model, see `loss_functions.py`.

### Record trained model, parameters and plots to `models` directory
On the second cell in `train_psi.ipynb`, there is a snippet of code that creates a directory each time you execute the notebook, this is intended to keep track of model parameters, save the loss landscape heatmap and loss graph produce by the model. If you don't need this feature, you can comment it out.

### Getting Started
You can install anaconda or miniconda to get the conda environment to manage the project's libraries.

After you cloned this repository, open the project in your preferred IDE and execute this command on your terminal:
```bash
conda create --name <env> --file requirements.txt
```
Where `<env>` is how you want to name your environment. This will create a new conda environment and install all the project dependencies.

Use this command to activate the environment you created.
```bash
conda activate <env> 
```

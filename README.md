# LatentSpaceBuilder

The Latent Space Builder builds the latent space for an image dataset. 

## Getting started

Clone this repository.

```bash
git clone https://github.com/compSPI/LatentSpaceBuilder.git
```

Change the current working directory to the root of this repository.

```bash
cd /path/to/LatentSpaceBuilder
```

Create the Python environment that has the dependencies required to run the code.

```bash
conda env create -f environment.yml
```

Activate the environment.
```bash
conda activate latent_space_builder
```

Install the kernel.
```bash
python -m ipykernel install --user --name latent_space_builder \ 
	--display-name "Python (latent_space_builder)"
```

Exit the environment.

```bash
conda deactivate
```

## Running the notebook

Run jupyter notebook.

```bash
jupyter notebook 
```

Open the tutorial notebook ```latent_space_builder.ipynb```.

Change the Python kernel to ```latent_space_builder```.

Set ```dataset_filepath``` to the file containing the dataset.

```python
dataset_filepath = '../data/cspi_synthetic_dataset_diffraction_patterns_1024x1040.hdf5'
```

Run the notebook.

## Code structure

The code for this repository is organized as follows:

- ```README.md```: Highlights the usefulness of the Latent Space builder. 

- ```latent_space_builder.ipynb```:  Provides a tutorial notebook for using the Latent Space builder.

- ```environment.yml```: Contains Python packages required to run the notebook.

- ```src/```: Contains Python files required to run the notebook.

- ```figures/```: Contains figures used in the repository.

# LatentSpaceBuilder

The Latent Space Builder builds the latent space for an image dataset. 

## Getting started

Clone this repository.

```bash
git clone https://github.com/compSPI/LatentSpaceBuilder.git
```

Change the current working directory to the root of this repository.

```bash
cd LatentSpaceBuilder
```

Download from the Anaconda Cloud and install the Python environment that has the dependencies required to run the code.

```bash
conda env create compSPI/compSPI
```

Activate the environment.

```bash
conda activate compSPI
```

Install the kernel.

```bash
python -m ipykernel install --user --name compSPI --display-name "Python (compSPI)"
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

Change the Python kernel to ```compSPI```.

Set ```dataset_file``` to an HDF5 file containing the dataset.

```python
dataset_file = '../data/cspi_synthetic_dataset_diffraction_patterns_1024x1040.hdf5'
```

Run the notebook.

## Installation

The project package can be installed by running the following command.

```bash
python setup.py install
```

## Code structure

The relevant files and folders in this repository are described below:

- ```README.md```: Highlights the usefulness of the Latent Space builder. 

- ```latent_space_builder.ipynb```:  Provides a tutorial notebook for using the Latent Space Builder.

- ```latent_space_builder/```: Contains the Python file required to run the notebook.

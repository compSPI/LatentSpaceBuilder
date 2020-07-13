# LatentSpaceBuilder

The Latent Space Builder builds the latent space for an image dataset. 

## Downloading the code

This command is used to download the code onto your computer.

```bash
git clone https://github.com/compSPI/LatentSpaceBuilder.git
```

## Installing the code onto your computer

This command is used to install the code onto your computer.

```bash
python setup.py install
```

## Installing the code in Jupyter Notebook

These commands are used to install the code onto Jupyter Notebook.

```bash
cd /path/to/LatentSpaceBuilder
conda env create compSPI/compSPI
conda activate compSPI
python -m ipykernel install --user --name compSPI --display-name "Python (compSPI)"
```

## Running the code in Jupyter Notebook

The following commands are used to run the code in Jupyter Notebook.

Start Jupyter Notebook.

```bash
jupyter notebook 
```

Open the tutorial notebook ```latent_space_builder.ipynb``` and change the Python kernel to ```compSPI```.

Find the following line in the notebook:

```python
dataset_file = '../data/cspi_synthetic_dataset_diffraction_patterns_1024x1040.hdf5'
```

Set the variable ```dataset_file``` to an HDF5 file containing the dataset.

Run the notebook.

## Code structure

The relevant files and folders in this repository are described below:

- ```README.md```: Highlights the usefulness of the Latent Space builder. 

- ```latent_space_builder.ipynb```:  Provides a tutorial notebook for using the Latent Space Builder.

- ```latent_space_builder/```: Contains the Python file required to run the notebook.

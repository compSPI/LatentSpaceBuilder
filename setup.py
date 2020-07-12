import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'bokeh>=2.1.1',
    'jupyterlab>=2.1.5',
    'scikit-learn>=0.23.1',
    'h5py>=2.10.0',
    'numpy>=1.19.0',
    'pyDiffMap>=0.2.0.1',
    'setuptools>=49.1.2'
]

setuptools.setup(
    name="latent-space-builder", # Replace with your own username
    version="0.0.1",
    author="Deeban Ramalingam",
    author_email="rdeeban@gmail.com",
    description="A tool for building latent spaces.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/compSPI/LatentSpaceBuilder",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
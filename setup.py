import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'setuptools',
    'numpy',
    'scikit-learn',
    'h5py',
    'pyDiffMap'
]

setuptools.setup(
    name="latent-space-builder",
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
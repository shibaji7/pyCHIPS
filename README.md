# ISWAT - International Space Weather Action Teams
## Welcome to Coronal Hole Boundary Detection using CHIPS Algorithm

This project is a part of [COSPAR ISWAT](https://www.iswat-cospar.org/iswat-cospar). The COSPAR ISWAT initiative is a global hub for collaborations addressing challenges across the field of space weather.


### The objectives of this work are following: 
- First, to build an `openCV` based automated coronal hole detection schemes. 
- Second, to develop strategies to quantitatively assess the spatial and temporal uncertainty of coronal hole boundary locations. 
- Third, to use this information to further improve the predictive capabilities of ambient solar wind models.

### Installation
`pyCHIPS` includes both a reusuable Python package (intended to be imported and reused by other projects) and a set of scripts / notebooks using that package. How you should install `pyCHIPS` depends on how you intend to use it.

**If you want to run the interactive scripts / notebooks**, you should download this repository and install from it. **If you want to use the library in other projects**, you can either do that or just install the library from a package index or use `pip` installer.

### If you want to run the interactive notebooks

Make sure you have a recent Python version (*Python>=3.7*) and Jupyter installed.

Clone this repository into a suitable location on your computer:

`git clone https://github.com/shibaji7/pyCHIPS`

`cd pyCHIPS`

`pip install .`

You should now be able to open and run the notebooks within `scripts/` using Jupyter.

### If you want to use the library in other projects

#### From the Python Package Index:

The `pyCHIPS` package is distributed on the Python Package Index: https://pypi.org/project/pyCHIPS/

`pip install pyCHIPS`

#### From the git repository:

Follow the same instructions as installing the interactive notebooks.

## Folder Structure

- `chips` -  the python package
- `docs` - documentation for the codebase
- `test`, etc. - example analysis scripts and Jupyter notebooks

### Setup for Developers
- Download this Repository via git-clone.
- Make sure you have anaconda environment and conda running.
- Run `conda env create -f chips.yml` command to create `chips` environment.
- Run `ipython kernel install --user --name chips` to install ipython kernel.
- Run Ipython notebooks inside docs under `chips` envioronment.

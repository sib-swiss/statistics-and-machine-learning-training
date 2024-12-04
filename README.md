[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10406621.svg)](https://doi.org/10.5281/zenodo.10406621)



This is the github repository for the "statistics and machine learning for life science" course at [SIB](https://www.sib.swiss).

This course was designed to guide participants in the exploration of the concepts of statistical modelling,
and at the same time relate and contrast them with machine learning approaches when it comes to both classification and regression.

## pre-requisites

The course is targeted to life scientists who are already familiar with the Python programming language and who have basic knowledge on statistics.

In order to follow the course you need to have installed [python](https://www.python.org/) and [jupyter notebooks](https://www.jupyter.org/) (a web based notebook system for creating and sharing computational documents).

If you need help with the installation, you can refer to these [tips and instructions](https://github.com/sib-swiss/first-steps-with-python-training/blob/master/setting_up_your_environment.md)(NB: this links to another github repo).

In order to install dependenices run the following command:
```
pip install -r requirements.txt
```

This will ensure you have the following libraries installed:
 * [seaborn](https://seaborn.pydata.org/installing.html)
 * [statsmodels](https://www.statsmodels.org/stable/install.html)
 * [scikit-learn](https://scikit-learn.org/stable/install.html)

> NB: we will use several other libraries, such as numpy, scipy, matplotlib, or pandas, but they are pre-requisites of the 3 above, so they should be installed automatically if you install using pip or conda.


## course organization

The course is organized in several, numbered, jupyter notebooks, each corresponding to a chapter which interleaves theory, code demo, and exercises.

The course does not require any particular expertise with jupyter notebooks to be followed, but if it is the first time you encounter them we recommend this [gentle introduction](https://realpython.com/jupyter-notebook-introduction/).

 * [00_python_warmup.ipynb](00_python_warmup.ipynb) : provides a gentle warm-up to the basic usage of the libraries which are a pre-requisite for this course. You can use this notebook to assess your level before the course, or just as a way to get (re-)acquainted with these libraries.
 * [01_regression_Least_Square.ipynb](01_regression_Least_Square.ipynb) : learn linear regression using Ordinary Least Square.
 * [02_regression_GLM.ipynb](02_regression_GLM.ipynb) : we change perspective from OLS to maximum likelihood Generalized Linear Regression, with example using Poisson/log, and logistic regression.
 * [03_Machine_Learning.ipynb](03_Machine_Learning.ipynb) : building on the previous example, we introduce how Machine Learning procedures helps us build more generalizable models.

Solutions to each practical can be found in the [`solution/`](solutions/) folder and should be loadable directly in the jupyter notebook themselves.

Note also the `utils.py` file contains many utilitary function we use to visually showcase the effect of various ML algorithms' hyper-parameters.


## course material citation

Please cite as :

 * Duchemin, W. (2023, December 19). Statistics and Machine Learning in Life Science - 2023 edition. Zenodo. https://doi.org/10.5281/zenodo.10406621

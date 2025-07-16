<p>
   <img height="270" align="center" src="https://github.com/fanzhanglab/pyCellPhenoX/blob/main/logo/cellphenoX_logo_banner.png?raw=true"> 
</p>

![PyPI](https://img.shields.io/pypi/v/pyCellPhenoX.svg)
![Python Version](https://img.shields.io/pypi/pyversions/pyCellPhenoX)
[![License](https://img.shields.io/pypi/l/pyCellPhenoX)][license] 
![Read the documentation at https://pycellphenox.readthedocs.io/](https://img.shields.io/readthedocs/pycellphenox/latest.svg?label=Read%20the%20Docs)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Ffanzhanglab%2FpyCellPhenoX&label=%23Visits&labelColor=%23000000&countColor=%2337d67a&style=plastic)

## Why the forked version?
pyCellPhenoX is currently unpublished and might still be under development. Here were the problems I came across when trying to use the unforked version as of July 14, 2025:
1. The current code breaks with newest version of CNA, which makes the Neighborhood Abundance Matrix 
2. The current demo shows that the NAM is a cell by cell matrix even though the NAM produced from the code is actually a cell by neighborhood. Additionally, CNA automatically calculates PCs for the NAM that we can use.
3. CNA requires numpy <2.0 while pyCellPhenoX shows a requirement of numpy >2.0. Installing via pip automatically will uninstall numpy to replace it with the ">2.0" version, thereby breaking the NAM function.
4. Actual Code changes:
   1. CellPhenoX.py was adapted to work with Shap above and below version 0.39 (different output formats), AND to work with either Logistic Regression (LogReg) or Random Forest (Random_Forest)

## Installation Instructions
**github** ([link](https://github.com/RefinedScience/pyCellPhenoX)):
``` bash
# install pyCellPhenoX directly from the forked version on github
git clone git@github.com:RefinedScience/pyCellPhenoX.git
```

## Dependencies/ Requirements
You can create a conda environment that works for both CNA and pyCellPhenoX by running `conda create env -f environment.yml`. 
The requirements.txt refers to the OLD version requirements.txt but will not work for numpy and cna together.

## Getting Started...

Here, we introduce CellPhenoX, an eXplainable machine learning method to identify cell-specific phenotypes that influence clinical outcomes for single-cell data. CellPhenoX integrates robust classification models, explainable AI techniques, and a statistical covariate framework to generate interpretable, cell-specific scores that uncover cell populations associated with a clinical phenotype of interest.

<img width="100%" align="center" src="https://github.com/fanzhanglab/pyCellPhenoX/blob/main/media/CellPhenoX_fig1.png?raw=true">

> Figure 1. CellPhenoX leverages cell neighborhood co-abundance embeddings, Xi , across samples and clinical variable Y as inputs. By applying an adapted SHAP framework for classification models, CellPhenoX generates Interpretable Scores that quantify the contribution of each feature Xi, along with covariates  and interaction term Xi, to the prediction of a clinically relevant phenotype Y. The results are visualized at single-cell level, showcasing Interpretable Scores at low-dimensional space, correlated cell type annotations, and associated marker genes.


## Tutorials
For the newest version: Please refer to X for an exmaple of using pyCellPhenoX on the AML dataset.
For the original: Please see the [Command-line Reference] for details. Additonally, please see [Vignettes] on the documentation page. 

## API
pyCellPhenoX has four major functions which are apart of the object:
1. split_data() - Split the data into training, testing, and validation sets 
2. model_train_shap_values() - Train the model using nested cross validation strategy and generate shap values for each fold/CV repeat
3. get_shap_values() - Aggregate SHAP values for each sample
4. get_intepretable_score() - Calculate the interpretable score based on SHAP values. 

Additional major functions associated with pyCellPhenoX are:
1. marker_discovery() - Identify markers correlated with the discriminatory power of the Interpretable Score.
2. nonNegativeMatrixFactorization() - Perform non Negative Matrix Factorization (NMF)
3. preprocessing() - Prepare the data to be in the correct format for CellPhenoX
4. principleComponentAnalysis() - Perform Principle Component Analysis (PCA)

Each function has uniqure arguments, see our [documentation] for more information


## License
Distributed under the terms of the [MIT license][license],
_pyCellPhenoX_ is free and open source software.

### Code of Conduct
For more information please see [Code of Conduct](CODE_OF_CONDUCT.md) or [Code of Conduct Documentation]

### Contributing
For more information please see [Contributing](CONTRIBUTING.md) or [Contributing Documentation]

## Issues
If you encounter any problems, please [file an issue] along with a detailed description. 

## Citation
If you have used `pyCellPhenoX` in your project, please use the citation below: 
</br>

 Young, J., Inamo, J., Caterer, Z., Krishna, R., Zhang, F. CellPhenoX: An eXplainable Cell-specific machine learning method to predict clinical Phenotypes using single-cell multi-omics, bioRxiv 2025.01.24.634132; doi: https://doi.org/10.1101/2025.01.24.634132

## Contact
Please contact [fanzhanglab@gmail.com](fanzhanglab@gmail.com) for
further questions or protential collaborative opportunities!

<!-- github-only -->

[license]: https://github.com/fanzhanglab/pyCellPhenoX/blob/main/LICENSE
[contributor guide]: https://github.com/fanzhanglab/pyCellPhenoX/blob/main/CONTRIBUTING.md
[file an issue]: https://github.com/fanzhanglab/pyCellPhenoX/issues/new
[command-line reference]: https://pycellphenox.readthedocs.io/en/latest/modules.html
[pipi]: https://pypi.org/project/pip/
[pypi]: https://pypi.org/project/pyCellPhenoX/
[vignettes]: https://pycellphenox.readthedocs.io/en/latest/vignettes/apply_cellphenoX_inflamed_uc_fibroblast.html
[documentation]: https://pycellphenox.readthedocs.io/
[Code of Conduct Documentation]: https://pycellphenox.readthedocs.io/en/latest/CODE_OF_CONDUCT.html
[Contributing Documentation]: https://pycellphenox.readthedocs.io/en/latest/CONTRIBUTING.html
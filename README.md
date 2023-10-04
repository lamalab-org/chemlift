<!--
<p align="center">
  <img src="https://github.com/lamalab-org/chemlift/raw/main/docs/source/logo.png" height="150">
</p>
-->

<h1 align="center">
  chemlift
</h1>

<p align="center">
    <a href="https://github.com/lamalab-org/chemlift/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/lamalab-org/chemlift/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/chemlift">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/chemlift" />
    </a>
    <a href="https://pypi.org/project/chemlift">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/chemlift" />
    </a>
    <a href="https://github.com/lamalab-org/chemlift/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/chemlift" />
    </a>
    <a href='https://chemlift.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/chemlift/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/lamalab-org/chemlift/branch/main">
        <img src="https://codecov.io/gh/lamalab-org/chemlift/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/lamalab-org/chemlift/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Chemical language interfaced predictions using large language models.

## 💪 Getting Started

With ChemLIFT you can use large language models to make predictions on chemical data. 
You can use two different approaches:

- **Few-shot learning**: Provide a few examples in the prompt along with the points you want to predict and the model will learn to predict the property of interest.
- **Fine-tuning**: Fine-tune a large language model on a dataset of your choice and use it to make predictions. 

Fine-tuning updates the weights of the model, while few-shot learning does not.

### Few-shot learning

```python
from chemlift.icl.fewshotclassifier import FewShotClassifier
from langchain.llms import OpenAI

llm = OpenAI()
fsc = FewShotClassifier(llm, property_name='bandgap')

# Train on a few examples
fsc.fit(['ethane', 'propane', 'butane'], [0,1,0])

# Predict on a few more
fsc.predict(['pentane', 'hexane', 'heptane'])
```

### Fine-tuning

```python

from chemlift.finetuning.classifier import ChemLIFTClassifierFactory

model = ChemLIFTClassifierFactory('property name',
                                    model_name='EleutherAI/pythia-1b-deduped').create_model()
model.fit(X, y)
model.predict(X)
```

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/chemlift/) with:

```shell
$ pip install chemlift
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/lamalab-org/chemlift.git
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/lamalab-org/chemlift/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.


### 📖 Citation

Citation goes here!

```
@article{Jablonka_2023,
    doi = {10.26434/chemrxiv-2023-fw8n4},
    url = {https://doi.org/10.26434%2Fchemrxiv-2023-fw8n4},
    year = 2023,
    month = {feb},
    publisher = {American Chemical Society ({ACS})},
    author = {Kevin Maik Jablonka and Philippe Schwaller and Andres Ortega-Guerrero and Berend Smit},
    title = {Is {GPT}-3 all you need for low-data discovery in chemistry?}
}
```



### 🎁 Support
The work of the LAMALab is supported by the Carl-Zeiss foundation. 

In addition, the work was supported by the MARVEL National Centre for Competence in Research funded by the Swiss National Science Foundation (grant agreement ID 51NF40-182892). In addition, we acknoweledge support by the USorb-DAC Project, which is funded by a grant from The Grantham Foundation for the Protection of the Environment to RMI’s climate tech accelerator program, Third Derivative. 



<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->


## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/lamalab-org/chemlift.git
$ cd chemlift
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/lamalab-org/chemlift/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/lamalab-org/chemlift.git
$ cd chemlift
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/chemlift/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>

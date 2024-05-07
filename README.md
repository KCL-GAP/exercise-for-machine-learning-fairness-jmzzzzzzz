# Fairness-Excercise
This repository is an assignment for the course "AI and Robotics Group Project" Week 10: Ethical and Trustworthy AI at King's College London. The assignment is about machine learning fairness measurement and mitigation. This repo provides the datasets and source code to help the students evaluate the fairness of the machine learning models and mitigate model bias using three existing common bias-mitigating methods.

## Task
The students need to install the experimental environment correctly and download the file from this repository to a local device to conduct the experiment. The coding task for you is to open the file "Measure_new" and fill in the three blanks (based on the document of IBM AIF360: https://aif360.res.ibm.com/) to calculate the three fairness metrics values.

## Experimental environment

Python version: >= Python 3.7 
Toolkit: IBM AI Fairness 360 (AIF360) 

Installation instructions for Python 3.7 and AIF360 can be found at https://github.com/Trusted-AI/AIF360. That page provides several ways to install it. We recommend creating a virtual environment for it (as shown below), because AIF360 requires specific versions of many Python packages which may conflict with other projects on your system. If you would like to try other installation methods or encounter any errors during the installation process, please refer to the page (https://github.com/Trusted-AI/AIF360) for help.

#### Conda

Conda is recommended for all configurations. [Miniconda](https://conda.io/miniconda.html)
is sufficient (see [the difference between Anaconda and
Miniconda](https://conda.io/docs/user-guide/install/download.html#anaconda-or-miniconda)
if you are curious) if you do not already have conda installed.

Then, to create a new Python 3.7 environment, run:

```bash
conda create --name aif360 python=3.7
conda activate aif360
```

The shell should now look like `(aif360) $`. To deactivate the environment, run:

```bash
(aif360)$ conda deactivate
```

The prompt will return to `$ `.

Note: Older versions of conda may use `source activate aif360` and `source
deactivate` (`activate aif360` and `deactivate` on Windows).

### Install with `pip`

To install the latest stable version from PyPI, run:

```bash
pip install 'aif360'
```

[comment]: <> (This toolkit can be installed as follows:)

[comment]: <> (```)

[comment]: <> (pip install aif360)

[comment]: <> (```)

[comment]: <> (More information on installing AIF360 can be found on https://github.com/Trusted-AI/AIF360.)

In addition, we require the following Python packages. 
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install --upgrade protobuf==3.20.0
pip install fairlearn
```

## Dataset

Please use the three benchmark datasets (e.g., Adult Census Income, German Credit, and Bank Marketing datasets) supported by the AIF360 toolkit. **When running the scripts that invoke these datasets, you will be prompted how to download these datasets and in which folders they need to be placed.** You can also refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Step-by-step Guide
You can also reproduce the results from scratch. We provide a step-by-step guide on how to reproduce the intermediate results.

We can obtain the original ML performance and fairness metric values by running `default.py.` `default.py` supports three arguments: `-d` configures the dataset; `-c` configures the ML algorithm; `-p` configures the protected attribute.
```
cd scouce_code_path
python default.py -d adult -c lr -p sex
python default.py -d adult -c lr -p race
python default.py -d german -c lr -p sex
python default.py -d german -c lr -p age
python default.py -d bank -c lr -p age

python maat.py -d adult -c lr -p sex
python maat.py -d adult -c lr -p race
python maat.py -d german -c lr -p sex
python maat.py -d german -c lr -p age
python maat.py -d bank -c lr -p age

python mirror_fair.py -d adult -c lr -p sex
python mirror_fair.py -d adult -c lr -p race
python mirror_fair.py -d german -c lr -p sex
python mirror_fair.py -d german -c lr -p age
python mirror_fair.py -d bank -c lr -p age

python rw.py -d adult -c lr -p sex
python rw.py -d adult -c lr -p race
python rw.py -d german -c lr -p sex
python rw.py -d german -c lr -p age
python rw.py -d bank -c lr -p age

python default.py -d adult -c rf -p sex
python default.py -d adult -c rf -p race
python default.py -d german -c rf -p sex
python default.py -d german -c rf -p age
python default.py -d bank -c rf -p age

python maat.py -d adult -c rf -p sex
python maat.py -d adult -c rf -p race
python maat.py -d german -c rf -p sex
python maat.py -d german -c rf -p age
python maat.py -d bank -c rf -p age

python mirror_fair.py -d adult -c rf -p sex
python mirror_fair.py -d adult -c rf -p race
python mirror_fair.py -d german -c rf -p sex
python mirror_fair.py -d german -c rf -p age
python mirror_fair.py -d bank -c rf -p age

python rw.py -d adult -c rf -p sex
python rw.py -d adult -c rf -p race
python rw.py -d german -c rf -p sex
python rw.py -d german -c rf -p age
python rw.py -d bank -c rf -p age

python default.py -d adult -c svm -p sex
python default.py -d adult -c svm -p race
python default.py -d german -c svm -p sex
python default.py -d german -c svm -p age
python default.py -d bank -c svm -p age

python maat.py -d adult -c svm -p sex
python maat.py -d adult -c svm -p race
python maat.py -d german -c svm -p sex
python maat.py -d german -c svm -p age
python maat.py -d bank -c svm -p age

python mirror_fair.py -d adult -c svm -p sex
python mirror_fair.py -d adult -c svm -p race
python mirror_fair.py -d german -c svm -p sex
python mirror_fair.py -d german -c svm -p age
python mirror_fair.py -d bank -c svm -p age

python rw.py -d adult -c svm -p sex
python rw.py -d adult -c svm -p race
python rw.py -d german -c svm -p sex
python rw.py -d german -c svm -p age
python rw.py -d bank -c svm -p age

```
### About the Results
After these steps, you can obtain the results of 15 (dataset, protected attribute, ML algorithm) combinations. The result for each combination is included in the `results/` folder. For example, in this folder, `maat_lr_adult_sex.txt` contains the results of MAAT for the (adult, sex, lr) combination. Each file in the folder has 53 columns, with the first column indicating the ML performance or fairness metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.


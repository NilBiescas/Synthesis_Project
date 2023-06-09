# Instant -> Traducción

<img src="https://www.careerguide.com/career/wp-content/uploads/2021/06/Translation-GIF.gif" width="450" height="250" />


Welcome to our project! We are Neil de la Fuente, Nil Biescas, Xavier Soto, Jordi Longaron, and Daniel Vidal, and we have joined forces to revolutionize the way [iDisc](https://www.idisc.com/en/), a translation company, assigns tasks to its translators.

  
## Table of Contents

- [Project Overview](#Project-Overview)
- [Repository Structure](#Repository-Structure)
- [Data](#Data)
- [Installation and Usage](#Installation-and-Usage)
- [Performance](#Performance)
- [How to Contribute](#How-to-Contribute)
- [Want to know more?](#Want-to-know-more?)
- [Contact](#Contact)

## Project Overview

Our mission is to assist project managers at iDisc in making task assignments more efficient and effective. To achieve this, we have developed several machine learning models, including a Random Forest with Decision Trees and a Multilayer Perceptron (MLP). These models take into account various factors such as previous tasks completed by translators, client preferences, and features of the task at hand. The output is a list of top-k candidates best suited for a given task, making the assignment process streamlined and informed.

## Repository Structure

- `Decision_Trees`: This directory contains Jupyter notebooks for the models we've built using decision trees. The notebooks included are "DecisionTrees_synthesis.ipynb" and "randomforest_synthesis.ipynb".
- `Models`: This directory contains the model used to trained the MLP.
- `CheckPoints`: This directory contains checkpoint files of the different models we've experimented with, each having its unique configuration, such as batch sizes and the use of dropout techniques.
- `Utils`: Inside this directory you will find three files:
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. **Utils.py**          used to obtain the dataloader  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. **organaizer.py**     used to organize the training and validation of the model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. **utils_Dataset.py**  used to preprocess all the data from the .pkl file  
- `TKinter`: This directory contains a python file using tkinter to create the interface of the project. For the in depth explanation access the folder.

## Data

Here you have a link for the data needed for each of the models (Might be different data due to the difference between decision trees and neural networks):
- [Data for Decision Trees and Random Forest](https://drive.google.com/drive/folders/1rRwvEvHWddtyI-3mC2S8FqJHDPvdnrBc?usp=sharing)
- [Data for MLP](https://drive.google.com/file/d/1HXp16KdiZmQc178FeFeSk_tYk7fsfDgk/view?usp=sharing)

Before the data is fed into our models, it undergoes a thorough preprocessing. This includes cleaning, normalization, and feature extraction, ensuring that our models receive quality data that helps them make the best predictions.

## Installation and Usage

Before starting with the usage, ensure Python 3.x is installed on your system. If it is not, you can download it [here](https://www.python.org/downloads/). Next, clone the project from GitHub to your local machine using the command:

```
git clone https://github.com/NilBiescas/Synthesis_Project.git
```

### Executing the MLP

To run the program you will need to do update the path to the data downloaded for the MLP. The variable that will need to be changed is found in the main.py file and it is name **pkl_file**.

```
python main.py
```

### Executing the Trees and Random Forest

Just download the notebooks, upload the data and run all the cells, yes, it´s that easy!

## Performance

Our models have shown promising results in optimizing the task assignment process. The Random Forest model and the MLP model achieved the following performance:

| **Model** | **Accuracy** | **Recall** | **F1-Score** |
| --------- | ---------- | ------------ | ------------ |
| Random Forest | 82% | 79% | 80% |
| MLP | 84% | 81% | 81% |

 We continue to improve and optimize these models. The performance measures are based on the accuracy of the task assignment.

## How to Contribute

We welcome contributions! If you're interested in improving our models, fixing bugs, or adding new features, please feel free to make a pull request.

## Want to know more?

Soon the report on the project will be available for you to have a deeper understanding of our work. Stay tuned for updates!


## Contact

For any inquiries or issues, feel free to reach out to us:

- [Neil de la Fuente](https://www.linkedin.com/in/neil-de-la-fuente)
- [Nil Biescas](https://www.linkedin.com/in/nil-biescas-rue-3b830b238/)
- [Jordi Longaron](jordilongaroncarbonell@gmail.com)
- [Xavi Soto](xaviminisoto@gmail.com)
- [Danie Vidal](https://www.linkedin.com/in/daniel-alejandro-vidal-guerra-21386b266/)


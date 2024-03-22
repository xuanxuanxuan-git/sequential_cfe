# RObust SEquential (ROSE) recourse generator

Our method generates a sequence of robust recourse, such that even if users implement the recourse in a noisy way, they are still very likely to get the desirable prediction outcome.

## Updated Experimental Results -- In response to Reviewer \#3 \& \#4

<img src="updated_experimental_results.PNG" width="650">

We have included the performance results of CROCO -- proposed in paper [a] (Generating robust counterfactual explanations. Guyomard, V., Fessant, F., Guyet, T., Bouadi, T. and Termier, A., 2023, September. In ECML PKDD). **The results do not change our conclusions**.


## Installation

Run the following command to install all the required packages to run ROSE
```
pip install -r requirements.txt
```
We use python 3.7 to run all of the experiments. We encourage you to use the same python version for maximum compatibility. 

## Experiments
To train the policy for each dataset, run the following command:
```
bash train_commands.sh
```
This file contains all the hyper-parameters we used to train our method ROSE.

To evaluate ROSE, run the following command:
```
bash evaluation_commands.sh
```

## Saved Results
The figures used in the paper, as well as the recourse generated by different methods, are all saved in ```/stored_variables/```. 

## Results
Experiment results and figures included in the paper are produced in the following files

```tree
results
├── figures.ipynb: experiment results on synthetic datasets.
├── noise_summary_figure.ipynb: pictorial representations of different ways to model noise.
├── lr_run.ipynb: run all recourse generators with a logistic regression predictor.
├── nn_run.ipynb: run all recourse generators with a neural network predictor.
├── noise_evaluation.py: evaluate the performance of all methods.
```

## Baselines

Wachter, GrSp, DiCE, CoGS, PROBE, ARAR, and ROAR are implemented using the [CARLA](https://github.com/carla-recourse/CARLA/tree/main?tab=readme-ov-file) package. We keep the implementation of the aforementioned baselines in a separate [code repository](https://anonymous.4open.science/r/robust_cfe_baselines) as the code environment requires packages of different versions. Instructions about this repository can be found in its [README.md](https://anonymous.4open.science/r/robust_cfe_baselines/README.md) file. 

To set up the environment for CARLA, run the following command:
```
pip install carla-recourse
``` 
CoGS, PROBE, ARAR, and ROAR are not implemented in the original CARLA package, we adapted these methods into the CARLA framework through the following reference:
- PROCE, ARAR, and ROAR: [source](https://github.com/MartinPawel/ProbabilisticallyRobustRecourse/tree/main)
- CoGS: [source](https://github.com/marcovirgolin/robust-counterfactuals)


FastAR can be run without additional configurations. 
- FastAR: [source](https://github.com/vsahil/FastAR-RL-for-generating-AR/tree/submit)


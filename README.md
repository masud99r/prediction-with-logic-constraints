# Sequential-Prediction-with-Logic-Constraints
Code for paper: Sequential Prediction with Logic Constraints

## File directory setup

`Sequential-Prediction-with-Logic-Constraints/`

`logic_constraints/`

"Sequential-Prediction-with-Logic-Constraints" contains source code and "logic_constraints" contain data. Assume that your are at the "Sequential-Prediction-with-Logic-Constraints" directory.

## Setup environment:
Ubuntu 18.04 and latest.
Run following commands in order to setup the environment.
Download anaconda3 from here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

`conda create -n mlnconstraint python=3.7`

`conda activate mlnconstraint`

`pip install torch torchvision`

`pip install -U scikit-learn`

`pip install pracmln`

-----------------
Assuming you are using the preprocessed data and learned constraints in the data folder "logic_constraints", you can run the following steps to generate results. 

## Generate Results:
You can change the configuration and run differetn variation of experiments by changing line 20-25 in the file `lstm.py`.

Set `ROBOT` to "Gym" or "Taurus" to generate results on Gym or Taurus dataset.

If `iscombined` is True then conflation is applied after the lstm prediction, otherwise the results would be without conflation.

The `early_frac` is the fraction of frames used for early prediction. The results reported in the paper with this value set to 0.5. Please change this value (0,1) if you want to see results on other fraction.

The `total_epoch` is total number of epoch to train. 

To generate figures and results, after completing the above steps, run `figures.py`. The results can be found in the folder "Sequential-Prediction-with-Logic-Constraints/results". For each method, change the robot "Gym" or "Taurus" to gemerate figure and compiled results for each robot.

-----------------

If you want to run the data preprocessing and constraint learning then run following steps. After these steps you can run the "Generate Results" step, to get new results of your configuration.

## Preprocess Data:
Run `preprocess.py` to preproces data for Taurus and Gym robot.

## Generate constraints:
1. Run `extract_motion_feature.py` to process data for constraints generation for both robots.
2. Run `generate_constraint.py` to generate constrains in the format MLN requires for training.

## Learn constraints
Run `learn_constraints_weight.py` to learn constraints from training data using MLN. Change the robot at line 60 to learn for "Gym" or "Taurus". 

Note that this process might take some time depending on the computing machine. A common error is the memory exceed error. In this case, increasing to a larger memory often solve the problem. In our experiments we used a RAM size of 256 GB.


# Final Project CMPUT 644
## Detecting Adversarial Attacks on Federated Learning with Unbalanced Classes

## Intro
In this project we investigate the resilience of proposed poisoning attacks and detection systems in a variety of adversarial proposed in [1]. Our main contributions, in addition to reproducing the poisoning attacks and corresponding detection strategies are as follows:

* We proposed a smooth and incremental method to generate unbalanced datasets and evaluated the performance of FedAvg, and the proposed poisoning detection mechanisms on unbalanced datasets.
* We implemented adversaries with varying amounts of poison, and model the behavior of the proposed poisoning detection strategies.
* We conducted experiments with two distinct distance metrics for one of the detection algorithms, and for the first time presented experimental results on the performance of detection metrics.

[1] Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and
Seraphin Calo. 2019. _Analyzing federated learning through an ad-
versarial lens._ In International Conference on Machine Learning. PMLR,
634–643.

## Experiments:
1. Reproducing Malicious attacks: `flower_args.py`
2. Effect of Malicious Attack on Model Accuracy: `flower_args.py`
3. Detecting the stealthy malicious attack: The weights of `flower_args.py` are saved (directory not uploaded to repo as it is ~51GB)
    3.1 Validation accuracy-based detection: `utils/detection_algo_val_accuracy`
    3.2 Distance ranges-based detection: `DetectionStats/Step2_DataCrunching.ipynb`, `DetectionStats/Step3_Visualization.ipynb`
4. Accuracy versus class imbalance: `run_imbalance_v_accuracy.py`
5. Detecting stealthy attacks on unbalanced data: `flower_args.py` is run with different parameters. See `$python3 flower_args.py --help`
    5.1 Validation accuracy-based detection: `utils/detection_algo_val_accuracy`
    5.2 Distance ranges-based detection: `DetectionStats/Step2_DataCrunching.ipynb`, `DetectionStats/Step3_Visualization.ipynb`
6. Summary of threshold values: `DetectuibStats/Step4_SummarizeKappa.ipynb`

## Dependencies:
The dependencies can be checked on `requirements.txt`. Using a `venv` is recommended.

## Acknowledgments:
We'd like to thank our Instructor and Supervisor for his wonderful teachings and guidance during this course. 

## Running on Snowball notes:
To use GPU 1 or 2, instead of 0 on snowball. Do `export CUDA_VISIBLE_DEVICES=1,2`  on the command line before running python script.

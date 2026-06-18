# Potential subgraph rule and reasoning context enhancement for sparse multi-hop knowledge graph reasoning

Our implementation is based on official codes of [PSAgent](https://github.com/rubickkcibur/PSAgent) and [DacKGR](https://github.com/THU-KEG/DacKGR). Best regards to their contribution.

## Quick Start

#### Mannually set up 
```
conda install --yes --file requirements.txt
```
### Train models
Then the following commands can be used to train the proposed models in the paper. By default, dev set evaluation results will be printed when training terminates.

```
./experiment-rs.sh configs/<dataset>-conf.sh --train <gpu-ID>
./experiment-rs.sh configs/nell23k-conf.sh --train 0 
```

* Note: To train the PreKGR models, make sure 1) you have pre-trained the embedding-based models and 2) set the file path pointers to the pre-trained embedding-based models correctly (configs/nell23k-rs.sh), for example, conve_state_dict_path="model/NELL23K-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar".



### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).

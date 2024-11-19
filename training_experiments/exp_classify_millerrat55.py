"""
Classify all models on Miller's rat 55 simulated agents & original data.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

n_blocks = 800
# n_trials = 500
n_trials = 1200

for model_names in [
    # ['GRU_1', 'MXs'],
    # ['GRU_2', 'MXs'],
    # ['SGRU_1', 'MB0s'],
    # ['GRU_1', 'MB0s'],
    ['MFs', 'MB0s'],
]:
    suffix_exp_name = f'{model_names[0]}vs{model_names[1]}'
    if n_trials != 500: # 500 is the default value
        suffix_exp_name += f'_ntrials{n_trials}'
    behav_data_spec = {
        'misc': [],
    }
    for model_idx, model_name in enumerate(model_names):
        behav_data_spec['misc'].append(
                             {
                                 'dataset': 'SimAgent',
                                 'label': model_idx,
                                'agent_path': f'allagents_millerrat55_nblocks{n_blocks}_ntrials{n_trials}',
                                'agent_name': f'{model_name}_seed0',
                             },
        )

    base_config = {
          ### dataset info
          'dataset': 'Misc',
          'behav_format': 'tensor',
          'behav_data_spec': behav_data_spec,
          ### model info
          'agent_type': 'RNN',
          'rnn_type': 'GRU', # which rnn layer to use
          'input_dim': 3,
          'hidden_dim': 10, # dimension of this rnn layer
          'output_dim': 2,
          'output_layer_num': 2, # number of output layers
          'output_dim1': len(model_names), # dimension of model labels
          'device': 'cuda',
          'output_h0': True, # whether initial hidden state included in loss
          'trainable_h0': False, # the agent's initial hidden state trainable or not
          'readout_FC': True, # whether the readout layer is full connected or not
          'one_hot': False, # whether the data input is one-hot or not
          ### training info for one model
          'label_prop': 0.5, # coefficient of model label loss
          'lr':0.005,
          'l1_weight': 1e-5,
          'weight_decay': 0,
          'penalized_weight': 'rec',
          'max_epoch_num': 20000,
          'early_stop_counter': 1000,
          'grad_clip': 1,
          ### training info for many models on dataset
          'single_inner_fold': True,
          'outer_splits': 5,
          'inner_splits': 4,
          'seed_num': 1,
          ### additional training info
          'save_model_pass': 'full', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
          'training_diagnose': ['plot_loss'], # can be a list of diagnose function strings
          ### current training exp path
          'exp_folder': get_training_exp_folder_name(__file__) + suffix_exp_name,
    }

    config_ranges = { # keys are used to generate model names
          'rnn_type': ['GRU'],
          'hidden_dim': [#5,
                         10,#20,
                         # 50
                         ],
          'l1_weight': [1e-5],
    }


    behavior_cv_training_config_combination(base_config, config_ranges)

"""
Run all models on monkey V, with neuro data as input
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'where', 'block_truncation': (10, 70)},
      'neuro_data_spec': {'select_bins': [2], 'preprocess':{'zcore': True, 'remove_nan': True, 'pca': 0}},
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cpu',
      'concat_neural_input': True,
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 2000,
      'early_stop_counter': 200,
      ### training info for many models on dataset
      'outer_splits': 5,
      'inner_splits': 4,
      'seed_num': 3,
      'single_inner_fold': True,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

for pca_num in [0,1,2,3]:
    base_config['neuro_data_spec']['preprocess']['pca'] = pca_num
    base_config['input_dim'] = 3 + pca_num
    config_ranges = { # keys are used to generate model names
          'rnn_type': ['GRU'],
          'hidden_dim': [#20,
                         2
                         ],
          'readout_FC': [True],
          'l1_weight': [1e-5],
          'pca_num': [pca_num],
    }
    # resource_dict = {'memory': 5, 'cpu': 16, 'gpu': 0}
    # #behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
    behavior_cv_training_config_combination(base_config, config_ranges)
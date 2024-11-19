"""
Run all models on Akam's rat (mouse) 49 simulated agents.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

base_config = {
      ### dataset info
      'dataset': 'SimAgent',
      'behav_format': 'tensor',
      'behav_data_spec': ['agent_path', 'agent_name'],
      'agent_path': ['allagents_akamrat49_nblocks100_ntrials150'],
      'agent_name': 'MFs_seed0',
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cuda',
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
      'outer_splits': 10,
      'inner_splits': 9,
      'single_inner_fold': True,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'agent_name': ['Q(1)_seed0',
                      'MFs_seed0',
                      'MF_bs_rb_ck_seed0',
                      'MB_bs_rb_ck_seed0',
                      'MF_dec_bs_rb_ck_seed0',
                      'MB_dec_bs_rb_ck_seed0',
                     ],
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,3,4
            #50, 20,10,5
                     ],
      'readout_FC': [True],
      'l1_weight': [1e-5,
                    #1e-4
                    ],
}
base_config.update({'device': 'cuda'})
# behavior_cv_training_job_combination(base_config, config_ranges, {'memory': 12, 'cpu': 1, 'gpu': 1})
behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
base_config.update({
      'input_dim': 8,
      'one_hot': True,
      'device': 'cuda',
})

config_ranges.update({
      'rnn_type': ['SGRU'],
      'hidden_dim': [1,],
})

# resource_dict = {'memory': 12, 'cpu': 16, 'gpu': 0}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
base_config = {
      ### dataset info
      'dataset': 'SimAgent',
      'behav_format': 'cog_session',
      'behav_data_spec': ['agent_path', 'agent_name'],
      'agent_path': ['allagents_akamrat49_nblocks100_ntrials150'],
      'agent_name': 'MFs_seed0',
      # 'both' for all blocks
      ### model info
      'agent_type': 'NTSCog',
      'cog_type': 'MB0',
      'device': 'cpu',
      ### training info for one model
      ### training info for many models on dataset
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = {  # keys are also used to generate model names
      'agent_name': [
                      'MF_bs_rb_ck_seed0',
                      'MB_bs_rb_ck_seed0',
                      'MF_dec_bs_rb_ck_seed0',
                      'MB_dec_bs_rb_ck_seed0',
                        'Q(1)_seed0',
                      'MFs_seed0',
                     ],
      'cog_type': [
                      'MF_bs_rb_ck',
                      'MB_bs_rb_ck',
                      'MF_dec_bs_rb_ck',
                      'MB_dec_bs_rb_ck',
                        'Q(1)',
                      'MFs',
      ],
}

# if __name__ ==  '__main__' or '.' in __name__:
#       behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)

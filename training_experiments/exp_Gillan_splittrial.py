"""
Run all models on Gillan Human. The train-val split is on trials, not subjects.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *
from utils import goto_root_dir
goto_root_dir.run()
from utils import set_os_path_auto

dt = Dataset('GillanHuman', behav_data_spec={}, verbose=False)
dt_aug = Dataset('GillanHuman', behav_data_spec={'augment': True}, verbose=False)
train_trial_index, val_trial_index, test_trial_index = split_over_trials(200, n_trial_splits=[150,25,25], seed=0)


# teacher network
base_config = {
      ### dataset info
      'dataset': 'GillanHuman',
      'behav_format': 'tensor',
      'behav_data_spec': {'augment': True},
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'include_embedding': False, # not useful for Gillan Human
      'input_dim': 10,
      'hidden_dim': 20, # dimension of this rnn layer
      'output_dim': 6, # dimension of action
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
      'max_epoch_num': 10000,
      'early_stop_counter': 200,
      'batch_size': 5000,
      ### training info for many models on dataset
      'split_training': True,
      'train_index': np.arange(0, dt_aug.batch_size),
        'val_index': np.arange(0, dt.batch_size),
        'test_index': np.arange(0, dt.batch_size),
      'train_trial_index': train_trial_index,
       'val_trial_index': val_trial_index,
      'test_trial_index': test_trial_index,
      'seed_num': 2,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': [], # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [
            #1,2,3,4,5,6,
            # 20,
            50, #NOTE: using batch_size= 5000

            #100,
            # 200,#400,500, 1000,
                     ],
      'l1_weight': [1e-5],
}

# behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
resource_dict = {'memory': 10, 'cpu': 1, 'gpu': 1}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)



# student RNN
run_student_rnn = False
if run_student_rnn:
      augment = True
      dt = Dataset('GillanHuman', behav_data_spec={'augment': augment,},verbose=False)
      teacher_test_indexes, teacher_model_path = load_teacher_summary('exp_Gillan_splittrial', selector={'hidden_dim': 50, 'distill': 'none'})

      # student RNN
      for test_index in teacher_test_indexes: # for each subject
            aug_test_index = dt.get_after_augmented_block_number([test_index])
            assert aug_test_index[0] == test_index
            base_config.update({
                  'behav_data_spec': {'augment': augment,},
                  'batch_size': 0,
                'train_index': aug_test_index,
                'val_index': aug_test_index,
                'test_index': aug_test_index[:1],
                  'train_trial_index': train_trial_index,
                   'val_trial_index': val_trial_index,
                  'test_trial_index': test_trial_index,
                'seed_num': 3,
                'teacher_model_path': teacher_model_path,
                  'split_training': True,
            })
            config_ranges = {
                  'block': [test_index],
                  'rnn_type': ['GRU'],
                'hidden_dim': [1,2,
                               3,4,5,20],
                'l1_weight': [1e-5,# 1e-4,
                              ],
                'distill': ['student'],
            }
            behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
            # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

            config_ranges.update({
                'hidden_dim': [3],
                'readout_FC': [False],
                'readout_block_num': [3],})
            behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
            # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
            config_ranges.pop('readout_FC', None)
            config_ranges.pop('readout_block_num', None)

run_cog = True
if run_cog and (__name__ == '__main__' or '.' in __name__):
      dt = Dataset('GillanHuman', behav_data_spec={}, verbose=False)
      # individual cognitive models
      for test_index in range(0, dt.batch_size): # for each subject
            base_cog_config = {
                  ### dataset info
                  'dataset': 'GillanHuman',
                  'behav_format': 'cog_session',
                  'behav_data_spec': {},
                  ### model info
                  'agent_type': 'OTSCog',
                  'cog_type': 'MF',
                  'device': 'cpu',
                  ### training info for many models on dataset
                  'split_training': True,
                  'train_index': [test_index],
                  'val_index': [test_index],
                  'test_index': [test_index],
                  'train_trial_index': train_trial_index,
                   'val_trial_index': val_trial_index,
                  'test_trial_index': test_trial_index,
                  'seed_num': 32,
                  ### additional training info
                  'save_model_pass': 'minimal',
                  # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
                  'training_diagnose': None,  # can be a list of diagnose function strings
                  ### current training exp path
                  'exp_folder': get_training_exp_folder_name(__file__),
            }

            config_cog_ranges = {  # keys are also used to generate model names
                  'block': [test_index],
                  'cog_type': ['MF', 'MB', 'MX',
                                'MFs', 'MBs', 'MXs',
                                'MFsr',
                               ],
            }
            behavior_cv_training_config_combination(base_cog_config, config_cog_ranges, n_jobs=-1, verbose_level=1)

run_student_cog = False
if run_student_cog and (__name__ == '__main__' or '.' in __name__):
      # student cog
      augment = 0
      teacher_test_indexes, teacher_model_path = load_teacher_summary('exp_Gillan_splittrial', selector={'hidden_dim': 50, 'distill': 'none'})
      # individual cognitive models
      for test_index in teacher_test_indexes:  # for each subject
            base_cog_config = {
                  ### dataset info
                  'dataset': 'GillanHuman',
                  'behav_format': 'cog_session',
                  'behav_data_spec': {},
                  ### model info
                  'agent_type': 'OTSCog',
                  'cog_type': 'MF',
                  'device': 'cpu',
                  ### training info for many models on dataset
                  'split_training': True,
                  'train_index': [test_index],
                  'val_index': [test_index],
                  'test_index': [test_index],
                  'train_trial_index': train_trial_index,
                  'val_trial_index': val_trial_index,
                  'test_trial_index': test_trial_index,
                  'seed_num': 32,
                 'teacher_model_path': teacher_model_path,
                  ### additional training info
                  'save_model_pass': 'minimal',
                  # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
                  'training_diagnose': None,  # can be a list of diagnose function strings
                  ### current training exp path
                  'exp_folder': get_training_exp_folder_name(__file__),
            }

            config_cog_ranges = {  # keys are also used to generate model names
                  'block': [test_index],
                  'cog_type': ['MF', 'MB', 'MX',
                               'MFs', 'MBs', 'MXs',
                               'MFsr',
                               ],
                  'distill': ['student'],
            }
            behavior_cv_training_config_combination(base_cog_config, config_cog_ranges, n_jobs=-1, verbose_level=1)

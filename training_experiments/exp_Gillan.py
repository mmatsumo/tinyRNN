"""
Run all models on Gillan Human.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *
from utils import goto_root_dir
goto_root_dir.run()
from utils import set_os_path_auto

# teacher network, should not be augmented
base_config = {
      ### dataset info
      'dataset': 'GillanHuman',
      'behav_format': 'tensor',
      'behav_data_spec': {},
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
      'batch_size': 100,
      ### training info for many models on dataset
      'outer_splits': 6,
      'inner_splits': 5,
      'single_inner_fold': True,
      'seed_num': 5,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': [], # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [
            1,2,3,4,5,6,
            # 20,50,#10,5,
            # 100,
            # 200,#400,500, 1000,
                     ],
      'l1_weight': [1e-5],
}

# behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
resource_dict = {'memory': 10, 'cpu': 1, 'gpu': 1}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)


# teacher RNN with blockwise readout
base_config.update({
      'readout_FC': False,
  })
config_ranges.update({
      'hidden_dim': [6, 3],
      'readout_block_num': [3],

})
# behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
base_config.update({
      'readout_FC': True, # reset
})
config_ranges.pop('readout_block_num', None)


# student RNN
run_student_rnn = False
if run_student_rnn:
      for k in ['outer_splits', 'inner_splits', 'single_inner_fold']:
              base_config.pop(k, None)
      with set_os_path_auto():
            trained_model_summary = joblib.load(ANA_SAVE_PATH / 'exp_Gillan' / f'rnn_final_best_summary.pkl')
            trained_model_summary = trained_model_summary[trained_model_summary['distill'] == 'none']

      teacher_summary = trained_model_summary[(trained_model_summary['hidden_dim'] == 100)]
      dt = Dataset('GillanHuman', behav_data_spec={'augment': True,},verbose=False)
      for outer_fold in pd.unique(teacher_summary['outer_fold']):
            this_outer_fold_summary = teacher_summary[teacher_summary['outer_fold'] == outer_fold]
            assert len(this_outer_fold_summary) == 1
            this_outer_fold_summary = this_outer_fold_summary.iloc[0]
            test_indexes = this_outer_fold_summary['test_index']
            teacher_model_path = this_outer_fold_summary['model_path']

            # student RNN
            for test_index in test_indexes: # for each subject
                  aug_test_index = dt.get_after_augmented_block_number([test_index])
                  assert aug_test_index[0] == test_index
                  base_config.update({
                        'behav_data_spec': {'augment': True,},
                        'batch_size': 0,
                      'train_index': aug_test_index[1:],
                      'val_index': aug_test_index[:1],
                      'test_index': aug_test_index[:1],
                      'seed_num': 3,
                      'teacher_model_path': teacher_model_path,
                        'split_training': True,
                  })
                  config_ranges = {
                        'block': [test_index],
                        'rnn_type': ['GRU'],
                      'hidden_dim': [#1,2,3,4,5,6,
                                     20],
                      'l1_weight': [1e-5,# 1e-4,
                                    ],
                      'distill': ['student'],
                  }
                  behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
                  # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)



if True and (__name__ == '__main__' or '.' in __name__):
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
                  'train_index': [test_index],
                  'val_index': [test_index],
                  'test_index': [test_index],
                  'seed_num': 16,
                  'split_training': True,
                  ### additional training info
                  'save_model_pass': 'minimal',
                  # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
                  'training_diagnose': None,  # can be a list of diagnose function strings
                  ### current training exp path
                  'exp_folder': get_training_exp_folder_name(__file__),
            }

            config_cog_ranges = {  # keys are also used to generate model names
                  'block': [test_index],
                  'cog_type': [#'MF', 'MB', 'MX'
                                # 'MFs', 'MBs', 'MXs',
                                'MFsr',
                               ],
            }
            behavior_cv_training_config_combination(base_cog_config, config_cog_ranges, n_jobs=-1, verbose_level=1)
"""
Run all models on Gillan Human. For RNNS with blockwise readout
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *
from utils import goto_root_dir
goto_root_dir.run()
from utils import set_os_path_auto

# teacher network
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
            20,50,#10,5,
            100,
            200,#400,500, 1000,
                     ],
      'l1_weight': [1e-5],
}

# behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
resource_dict = {'memory': 10, 'cpu': 1, 'gpu': 1}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

# student RNN
run_student_rnn = True
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

            def config_modifier_model_init(config):
                  # trained_model_summary and outer_fold are global variables
                  assert 'pretrained' in config and config['pretrained']
                  this_outer_fold_modelinit_summary = trained_model_summary[(trained_model_summary['outer_fold'] == outer_fold) &
                                                                      (trained_model_summary['hidden_dim'] == config['hidden_dim']) &
                                                                      (trained_model_summary['rnn_type'] == config['rnn_type']) &
                                                                      (trained_model_summary['readout_FC'] == config['readout_FC'])]
                  assert len(this_outer_fold_modelinit_summary) == 1
                  this_outer_fold_modelinit_summary = this_outer_fold_modelinit_summary.iloc[0]
                  model_init_path = this_outer_fold_modelinit_summary['model_path']
                  config['model_init_path'] = str(model_init_path)
                  return config

            # student RNN
            for test_index in test_indexes: # for each subject
                  aug_test_index = dt.get_after_augmented_block_number([test_index])
                  assert aug_test_index[0] == test_index
                  # unconstrained student RNN
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
                      'hidden_dim': [1,2,3,4,5,6],
                      'l1_weight': [1e-5,# 1e-4,
                                    ],
                      'distill': ['student'],
                  }

                  # student RNN with blockwise readout (GRU)
                  base_config.update({
                        'readout_FC': False,
                    })
                  config_ranges.update({
                        'hidden_dim': [6, 3],
                        'readout_block_num': [3],
                  })
                  # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
                  # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

                  # pretraining; currently not help here
                  config_ranges.update({
                        'pretrained': [True],
                  })
                  # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1, config_modifier=config_modifier_model_init)
                  behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, combined_yaml=True, config_modifier=config_modifier_model_init,
                                                       ignore_exist=True)
                  config_ranges.pop('pretrained', None) # reset

                  config_ranges.update({
                        'distill_temp': [1,2,4,8],
                        'teacher_prop': [0, 0.2, 0.8, 1],
                  })
                  # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
                  # behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, combined_yaml=True, ignore_exist=True)
                  config_ranges.pop('distill_temp', None)
                  config_ranges.pop('teacher_prop', None)

                  # student RNN with blockwise readout and MLR
                  base_config.update({
                        'device': 'cpu',
                  })
                  config_ranges.update({
                        'rnn_type': ['MLR'],
                        'nonlinearity': ['tanh'],
                        'l1_weight': [1e-5],
                        'expand_size': [40],
                  })

                  # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)
                  # behavior_cv_training_job_combination(base_config, config_ranges, {'memory': 8, 'cpu': 16, 'gpu': 0}, combined_yaml=True)

                  # reverse the changes
                  base_config['readout_FC'] = True
                  base_config['device'] = 'cuda'
                  config_ranges.pop('readout_block_num', None)
                  config_ranges.pop('expand_size', None)
                  config_ranges.pop('nonlinearity', None)





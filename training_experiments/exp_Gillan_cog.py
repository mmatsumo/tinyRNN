"""
Run all cognitive models on Gillan Human. These are trained on some subjects and evaluated on the unseen subjects.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *
from utils import goto_root_dir
goto_root_dir.run()
from utils import set_os_path_auto

if True and (__name__ == '__main__' or '.' in __name__):
      dt = Dataset('GillanHuman', behav_data_spec={}, verbose=False)
      # individual cognitive models
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
              'outer_splits': 6,
              'inner_splits': 5,
              'single_inner_fold': True,
              'seed_num': 16,
              ### additional training info
              'save_model_pass': 'minimal',
              # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
              'training_diagnose': None,  # can be a list of diagnose function strings
              ### current training exp path
              'exp_folder': get_training_exp_folder_name(__file__),
        }

      config_cog_ranges = {  # keys are also used to generate model names
              'cog_type': ['MF', 'MB', 'MX',
                            'MFs', 'MBs', 'MXs',
                            'MFsr',
                           ],
        }
      behavior_cv_training_config_combination(base_cog_config, config_cog_ranges, n_jobs=-1, verbose_level=1)
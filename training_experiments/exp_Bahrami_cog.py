"""
Run all cognitive models on Bahrami Human, four-armed bandit. These are trained on some subjects and evaluated on the unseen subjects.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

if True and (__name__ == '__main__' or '.' in __name__):
    dt = Dataset('BahramiHuman', behav_data_spec={}, verbose=False)
    # individual cognitive models
    base_cog_config = {
          ### dataset info
          'dataset': 'BahramiHuman',
          'behav_format': 'cog_session',
          'behav_data_spec': {},
          ### model info
          'agent_type': 'MABCog',
          'cog_type': 'MFD',
          'n_actions': 4,
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
          'cog_type': ['MF', 'MFD',
                'MFL',
                # 'MFLslr',
                'MFLabs',
                'MFDp',
                 ],
    }
    behavior_cv_training_config_combination(base_cog_config, config_cog_ranges, n_jobs=-1, verbose_level=1)
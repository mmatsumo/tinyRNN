import os

from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
import joblib

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    # 'copy_scores_for_each_exp_from_sim',
    # 'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'sym_regression_for_each_model',
]

exp_folders = [
    'exp_sim_millerrat55',
    # 'exp_sim_millerrat55_nblocks200',
]


## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                             additional_rnn_keys={'model_identifier_keys': ['agent_name']},
                             additional_cog_keys={'model_identifier_keys': ['agent_name']},
                                 include_acc=True,
                                 dataset_loading_every_time=True)# multiple datasets in one folder; must be re-loaded every time


# dynamics
if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder, pointwise_loss=True)

def copy_scores_from_sim(sim_folder, exp_folder):
    # iterate each file in the sim_folder
    sim_path = SIM_SAVE_PATH / sim_folder
    for file_name in os.listdir(sim_path):
        if file_name.endswith('.pkl') and 'config' not in file_name:
            model_name = file_name[:-4]
            config = joblib.load(sim_path / (model_name + '_config.pkl'))
            behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'], verbose=False)
            behav = joblib.load(sim_path / file_name) # dict with action, stage2, reward, params, mid_vars
            behav_dt._detect_trial_type(behav)
            b = {
                'scores': [mid_vars_block['scores'] for mid_vars_block in behav['mid_vars']],
                'internal': [],
                'trial_type': np.array(behav['trial_type']),
                'hid_state_lb': None,
                'hid_state_ub': None,
                'behav_loss': None,
                'mask': None,
                'augment_ratio': None,
            }

            ana_model_path = f'{exp_folder}/agent_name-{model_name}/outerfold0_innerfold0_seed0'
            print(ana_model_path)
            os.makedirs(ANA_SAVE_PATH / ana_model_path, exist_ok=True)
            joblib.dump(b, ANA_SAVE_PATH / ana_model_path / 'total_scores.pkl')

if 'copy_scores_for_each_exp_from_sim' in analyzing_pipeline:
    for sim_folder, exp_folder in [
        ('RTS_agents_millerrat55', 'exp_sim_millerrat55'),
    ]:
        copy_scores_from_sim(sim_folder, exp_folder)

if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_2d_inits_exp(exp_folder, grid_num=50)

if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder, model_identifier_keys=['agent_name', 'model_type'])

#sym regression
if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_sim_millerrat55\agent_name-LS0_seed0.rnn_type-SGRU.hidden_dim-1.readout_FC-True.l1_weight-1e-05\outerfold6_innerfold3_seed0',
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)
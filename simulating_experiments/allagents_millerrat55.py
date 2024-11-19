"""
Run simulations of the agents, with parameters fit to the Miller Rat 55 data.
This file overlaps with RTS_agents_millerrat55.
This file is mainly for generating the behavioral data that are directly fitted to different models.
800 blocks and 500 trials are usually enough.
"""
import sys
sys.path.append('..')
from tasks import akam_tasks as ts
from utils import *
from path_settings import *
from simulate_experiment import *
import joblib
goto_root_dir.run()

sim_config = {
    'task': 'Akam_RTS',
    'com_prob': 0.8, # common transition probability
    'rew_gen': 'blocks',
    'block_length': 50,
    'rew_probs':[0.2, 0.8],
    'n_blocks': 100,
    'n_trials': 100,
    'sim_seed': 0,
    'sim_exp_name': get_current_file_name(__file__),
    'additional_name': '',
    'use_one_hot': 'cso', # no effect for cog models and GRU models, but work for SGRU
}

ana_exp_path = ANA_SAVE_PATH / 'exp_seg_millerrat55'

sim_exp_name = get_current_file_name(__file__)

for n_blocks, n_trials in [
    # (800, 500),
    (800, 1200),
    ]:
    sim_config['n_blocks'] = n_blocks
    sim_config['n_trials'] = n_trials
    sim_config['sim_exp_name'] = sim_exp_name + '_nblocks' + str(n_blocks) + '_ntrials' + str(n_trials)

    task = ts.Two_step(com_prob=sim_config['com_prob'], rew_gen=sim_config['rew_gen'],
                     block_length=sim_config['block_length'],
                     probs=sim_config['rew_probs'])

    path = ana_exp_path / 'cog_final_best_summary.pkl'
    cog_summary = joblib.load(path)

    path = ana_exp_path / 'rnn_final_best_summary.pkl'
    rnn_summary = joblib.load(path)

    summary_cond = lambda summary: len(summary) == 10
    iter_model_infos = [] # (model_type, additional_name, this_model_summary, summary_cond)
    for model_type in [   #'MB0s', 'LS0', 'LS1', 'MB0', 'MB1', 'MB0md', 'RC', 'Q(0)', 'Q(1)',
                #'MXs',
            'MFs',
            ]:
        iter_model_infos.append((model_type, '', cog_summary[cog_summary['cog_type'] == model_type], summary_cond))
    # iter_model_infos.append(
    #     ('SGRU', '1', rnn_summary[
    #         (rnn_summary['rnn_type'] == 'SGRU') & (rnn_summary['hidden_dim'] == 1) & (rnn_summary['readout_FC'])],
    #      summary_cond))
    # iter_model_infos.append(
    #     ('GRU', '1', rnn_summary[
    #         (rnn_summary['rnn_type'] == 'GRU') & (rnn_summary['hidden_dim'] == 1) & (rnn_summary['readout_FC'])],
    #      summary_cond))
    # iter_model_infos.append(
    #     ('GRU', '2', rnn_summary[
    #         (rnn_summary['rnn_type'] == 'GRU') & (rnn_summary['hidden_dim'] == 2) & (rnn_summary['readout_FC'])],
    #      summary_cond))

    iterate_each_model(iter_model_infos, task, sim_config)
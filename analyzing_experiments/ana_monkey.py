from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    'compile_perf_for_all_exps',
    # 'extract_model_par',
    # 'logit_vs_action_freq',
    # 'action_freq_after_action_seq',
    # 'run_scores_for_each_exp',
    # 'run_scores_for_each_exp_best_for_test',
    # 'run_2d_inits_for_each_exp',
    # 'analyze_rt_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'sym_regression_for_each_model',
    # 'run_dynamical_regression_for_each_exp',
    # 'neural_decoding_for_each_model',
    # 'ccgp',
    # 'neural_dynamics_modeling_for_each_exp',
    # 'neural_decoding_two_model_compare',
    # 'analyze_model_perf_for_each_data_proportion',
    # 'run_scores_for_each_data_proportion',
    # 'analyze_markov_matrix_for_each_exp',
    # 'analyze_reversal_alignment_for_each_exp',
]

exp_folders = [
    'exp_monkeyV',
    'exp_monkeyW',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                                 additional_rnn_keys={'model_identifier_keys': ['complex_readout','symm','finetune',
                                                                                'nonlinearity','expand_size',
                                                                                ]},
                                 include_acc=True,
                                 )

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_monkey',
                          rnn_filter={'readout_FC': True,
                                      'symm': 'none',
                                      'finetune': 'none',
                                      'complex_readout': 'none'}
                          )

if 'extract_model_par' in analyzing_pipeline:
    extract_model_par(exp_folders[0])

if 'logit_vs_action_freq' in analyzing_pipeline:
    for exp_folder in exp_folders[:1]:
        logit_vs_action_freq(exp_folder)

if 'action_freq_after_action_seq' in analyzing_pipeline:
    for exp_folder in exp_folders:
        action_freq_after_action_seq(exp_folder)
# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder, pointwise_loss=True)
    if 'run_scores_for_each_exp_best_for_test' in analyzing_pipeline:
        run_scores_exp(exp_folder, best_for_test=True)
    if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
        run_2d_inits_exp(exp_folder, grid_num=50)

if 'analyze_rt_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        compare_logit_rt(exp_folder)

if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

#sym regression
if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_monkeyV\rnn_type-GRU.hidden_dim-2.readout_FC-True.l1_weight-0.0001\outerfold0_innerfold8_seed0',
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)


def extract_feature_func(kept_feat_idx, selected_bins=[2,#3,4
                     ]
                     ):
    neuron_num, bin_num = kept_feat_idx.shape
    # print(session_name, 'kept_feat_idx', kept_feat_idx.shape, np.sum(kept_feat_idx))
    subset_from_kept_feat = []
    for n in range(neuron_num):
        for bin in range(bin_num):
            if kept_feat_idx[n, bin]:
                if bin in selected_bins:
                    subset_from_kept_feat.append(True)
                else:
                    subset_from_kept_feat.append(False)
    subset_from_kept_feat = np.array(subset_from_kept_feat)
    return subset_from_kept_feat

# neural decoding; should be run after dynamics/scores analysis
if 'neural_decoding_for_each_model' in analyzing_pipeline:
    for exp_folder in exp_folders:
        # run_decoding_exp(exp_folder, {'block_type': 'where'},
        #                  analyses=[
        #                      # 'decode_logit',
        #                      'decode_logit_abs',
        #                      # 'decode_logit_ortho',
        #                      # 'decode_logit_change',
        #                      # 'decode_value',
        #                      # 'decode_chosen_value',
        #                  ])
        # run_decoding_exp(exp_folder, {'block_type': 'where'}, # to be predicted
        #                  analyses=[
        #                      'latent_decode_neuron',
        #                      'latent_value_decode_neuron',
        #                      # 'value_decode_neuron',
        #                      # 'task_var_decode_neuron',
        #                      # 'task_var_value_decode_neuron',
        #                  ])

        ## compile_decoding_results only used when use variables to decode neuronal activity
        # compile_decoding_results(exp_folder, {'block_type': 'where'}, extract_feature_func=extract_feature_func)

        filters = lambda row: ('cog_type' in row and row['cog_type'] in ['LS0', 'LS1', 'MB0s', 'MB1', 'MB0']) or \
                                 ('rnn_type' in row and row['rnn_type'] in ['SGRU','GRU'] and row['readout_FC'])
        for predictor_bin in [0,1,2,3,4,5]:
            for to_predicted_bin in [0,1,2,3,4,5]:
                # run_decoding_exp(exp_folder, {'block_type': 'where', 'select_bins': [to_predicted_bin]},  # to be predicted
                #                  predictor_neuro_data_spec={'block_type': 'where', 'select_bins': [predictor_bin]},  # as predictor
                #                  analyses=[
                #                      # 'latent_decode_neuron',
                #                      # 'latent_value_decode_neuron',
                #                      'latent_value_interact_decode_neuron',
                #                      # 'value_decode_neuron',
                #                      # 'task_var_decode_neuron',
                #                      # 'task_var_value_decode_neuron',
                #                  ],
                #                  filters=filters,
                #                  )
                compile_decoding_results(exp_folder, {'block_type': 'where', 'select_bins': [to_predicted_bin]},
                                     extract_feature_func=None,
                                         other_fname=f'_predictor_block_type_where_select_bins_[{predictor_bin}]',
                                         filters=filters,)

if 'ccgp' in analyzing_pipeline:
    for exp_folder in ['exp_monkeyV', 'exp_monkeyW'
                       ]:
        filters = lambda row: ('rnn_type' in row and row['rnn_type'] in ['SGRU'] and row['readout_FC'] and row['hidden_dim'] == 1)
        for bin in [0,1,2,3,4,5]:
            print('=====================', bin, '=====================')
            run_decoding_exp(exp_folder, {'block_type': 'where', 'select_bins': [bin]},
                             analyses=[
                                 # global controlling hyperparameter
                                 'prev_trial', # this affect the behavior of ccgp and condition_decoding
                                 'balanced_dichotomy', # this affect the behavior of ccgp and condition_decoding

                                 # real analysis
                                 'ccgp',
                                 'condition_decoding',
                                 # 'mds',
                             ],
                             filters=filters,
                             )

if 'neural_dynamics_modeling_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders[:1]:
        run_neural_dynamics_modeling_exp(exp_folder, {'block_type': 'where', 'select_bins': [2,3], 'combine_bins': True})


if 'neural_decoding_two_model_compare' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_two_model_compare_decoding(exp_folder, {'block_type': 'where'})


if 'run_dynamical_regression_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        # run_dynamical_regression_exp(exp_folder, func='bartolo_2d',
        #                              filters=lambda row:
        #                              ('cog_type' in row and row['cog_type'] in ['MB0', 'MB1']) or \
        #                              ('rnn_type' in row and row['rnn_type'] == 'GRU' and not row['readout_FC']),
        #                              )
        run_dynamical_regression_exp(exp_folder, func='bartolo_1d',
                                     filters=lambda row:
                                     ('cog_type' in row and row['cog_type'] in ['MB0s', 'LS0']) or \
                                     ('rnn_type' in row and row['rnn_type'] == 'SGRU' and row['hidden_dim'] == 1),
                                     )

exp_folders = [
    'exp_monkeyV_dataprop',
    # 'exp_monkeyW_dataprop',
]
if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
                             additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )

for exp_folder in exp_folders:
    if 'run_scores_for_each_data_proportion' in analyzing_pipeline:
        run_scores_exp(exp_folder)

def create_transition_graph(d):
    import graphviz
    g = graphviz.Digraph('transition_graph', format='png', engine='neato')

    # Nodes with pos attribute
    g.node('00', 'A1 R: 0', shape='rectangle', pos='0,3!')
    g.node('01', 'A1 R: 1', shape='rectangle', pos='0,0!')
    g.node('10', 'A2 R: 0', shape='rectangle', pos='4,3!')
    g.node('11', 'A2 R: 1', shape='rectangle', pos='4,0!')
    g.node('0', 'A1', shape='circle', pos='2,3!')
    g.node('1', 'A2', shape='circle', pos='2,0!')

    for (action, reward), next_actions in d.items():
        for next_action, probability in next_actions.items():
            intermediate_node = f"{action}{reward}"
            g.edge(intermediate_node, str(next_action), label=f"{probability:.2f}")

    return g

def analyze_markov_matrix_for_each_exp(animal_name):
    config = {
        ### dataset info
        'dataset': 'BartoloMonkey',
        'behav_format': 'tensor',
        'behav_data_spec': {'animal_name': animal_name, 'filter_block_type': 'both', 'block_truncation': (10, 70)},
    }
    behav_data_spec = config['behav_data_spec']
    dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec)
    from collections import defaultdict
    transition_count = defaultdict(lambda: defaultdict(int))
    for block in range(len(dt.behav['action'])):
        action = dt.behav['action'][block]
        reward = dt.behav['reward'][block]
        for t in range(len(action)-1):
            transition_count[(action[t], reward[t])][action[t+1]] += 1.0
    # to frequency

    for k, v in transition_count.items():
        total = sum(v.values())
        for k2, v2 in v.items():
            transition_count[k][k2] = v2/total
    # change to normal dict of dict
    transition_count = {k: dict(v) for k, v in transition_count.items()}
    print(transition_count)
    transition_graph = create_transition_graph(transition_count)
    transition_graph.render(f'transition_graph_{animal_name}', view=True)

if 'analyze_markov_matrix_for_each_exp' in analyzing_pipeline:
    analyze_markov_matrix_for_each_exp('V')

def analyze_reversal_alignment():
    act_list = []
    GRU_prob_list = []
    MF_prob_list = []
    for animal_name in ['V', 'W']:
        # load behav
        config = {
            ### dataset info
            'dataset': 'BartoloMonkey',
            'behav_format': 'tensor',
            'behav_data_spec': {'animal_name': animal_name, 'filter_block_type': 'both', 'block_truncation': (10, 70)},
        }
        behav_data_spec = config['behav_data_spec']
        dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec)
        behav = dt.behav
        if animal_name == 'V':
            GRU_scores = joblib.load(
                ANA_SAVE_PATH / r'exp_monkeyV\rnn_type-GRU.hidden_dim-2.readout_FC-False.l1_weight-0.0001\outerfold2_innerfold2_seed0\total_scores.pkl')['scores']
            MF_scores = joblib.load(
                ANA_SAVE_PATH / r'exp_monkeyV\cog_type-MB1\outerfold2_innerfold2_seed0\total_scores.pkl')['scores']
        else:
            GRU_scores = joblib.load(
                ANA_SAVE_PATH / r'exp_monkeyW\rnn_type-GRU.hidden_dim-2.readout_FC-False.l1_weight-0.0001\outerfold8_innerfold0_seed0\total_scores.pkl')['scores']
            MF_scores = joblib.load(
                ANA_SAVE_PATH / r'exp_monkeyW\cog_type-MB1\outerfold8_innerfold1_seed0\total_scores.pkl')['scores']
        # turn to prob with softmax

        for b in range(dt.batch_size):
            GRU_prob = np.exp(GRU_scores[b]) / np.exp(GRU_scores[b]).sum(axis=1, keepdims=True)
            MF_prob = np.exp(MF_scores[b]) / np.exp(MF_scores[b]).sum(axis=1, keepdims=True)
            reversal_trial = np.where(behav['reversal_trial'][b])[0][0]
            assert 20<=reversal_trial<=40
            act_before = behav['action'][b][:reversal_trial]
            act_after = behav['action'][b][reversal_trial:]
            # print('reversal_trial', reversal_trial,'act_before', act_before.mean(), 'act_after', act_after.mean())

            if np.mean(act_before) < np.mean(act_after):
                act_before = 1 - act_before
                act_after = 1 - act_after
                GRU_prob = 1 - GRU_prob
                MF_prob = 1 - MF_prob
            # now action before should be ~1, after should be ~0
            act = np.concatenate([act_before[-20:], act_after[:20]])
            act_list.append(act)
            GRU_prob_list.append(GRU_prob[reversal_trial-20:reversal_trial+20,1])
            MF_prob_list.append(MF_prob[reversal_trial-20:reversal_trial+20,1])

    act_all = np.array(act_list)

    act_all_mean = np.mean(act_all, axis=0)
    # 95% CI
    act_all_ci = 1.96 * np.std(act_all, axis=0) / np.sqrt(len(act_all))
    from plotting_experiments.plotting import plot_start
    fig, ax = plot_start()
    plt.plot(np.arange(-20,20), act_all_mean, label='Data',color='C6')
    plt.fill_between(np.arange(-20,20), act_all_mean-act_all_ci, act_all_mean+act_all_ci, alpha=0.3, color='C6')
    plt.plot(np.arange(-20,20), np.mean(GRU_prob_list, axis=0), label='GRU',color='C0')
    # plt.plot(np.arange(-20,20), np.mean(MF_prob_list, axis=0), label='Model-free', color='C4')

    plt.vlines(0, 0, 1, linestyles='dashed', color='k', alpha=0.5)
    plt.xlim(-20,20)
    plt.ylim(0,1)
    plt.xticks([-20, 0,20])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('Trials from reversal')
    plt.ylabel('P(best initial)')
    plt.legend()
    plt.savefig(FIG_PATH / 'exp_monkey' / f'reversal_alignment_action_prob.pdf', bbox_inches='tight')
    plt.close()
    # plt.show()

if 'analyze_reversal_alignment_for_each_exp' in analyzing_pipeline:
    analyze_reversal_alignment()

from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    'compile_perf_for_all_exps',
    # 'extract_model_par',
    # 'run_scores_for_each_exp_best_for_test',
    # 'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'extract_ev_for_each_exp',
    # 'sym_regression_for_each_model',
    # 'neural_decoding_for_each_model',
    # 'analyze_model_perf_for_each_data_proportion',
    # 'run_scores_for_each_data_proportion',
    # 'analyze_stay_prob_for_each_exp',
]
exp_folders = [
    'exp_seg_millerrat55',
    'exp_seg_millerrat64',
    'exp_seg_millerrat70',
    'exp_seg_millerrat71',
]

## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                                 additional_rnn_keys={'model_identifier_keys': ['symm','finetune']},
                                 include_acc=True,
                                 )

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_seg_millerrat',
                          rnn_filter={'readout_FC': True,
                                      'symm': 'none'}
                          )


if 'extract_model_par' in analyzing_pipeline:
        extract_model_par(exp_folders[0])

if 'extract_ev_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_sv_for_exp(exp_folder)
# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder, pointwise_loss=True)
    if 'run_scores_for_each_exp_best_for_test' in analyzing_pipeline:
        run_scores_exp(exp_folder, best_for_test=True)
    if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
        run_2d_inits_exp(exp_folder, grid_num=50)


# 1d logit features
if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

#sym regression
if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_seg_millerrat55\rnn_type-SGRU.hidden_dim-2.readout_FC-False.l1_weight-0.0001\outerfold0_innerfold8_seed1'
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)

def extract_feature_func(kept_feat_idx):
    n_neurons, n_events, bin_num = kept_feat_idx.shape
    # print('kept_feat_idx', kept_feat_idx.shape, np.sum(kept_feat_idx))

    subset_from_kept_feat = []
    for n in range(n_neurons):
        for e in range(n_events):
            for bin in range(bin_num):
                if kept_feat_idx[n, e, bin]:
                    if (e == 2 and 10<=bin<=20) or (e == 3 and 5<=bin<15):
                        #if e in [2, 3] and 5<=bin<15:
                        subset_from_kept_feat.append(True)
                    else:
                        subset_from_kept_feat.append(False)
    subset_from_kept_feat = np.array(subset_from_kept_feat)
    return subset_from_kept_feat


# neural decoding; should be run after dynamics/scores analysis
if 'neural_decoding_for_each_model' in analyzing_pipeline:
    for exp_folder in exp_folders:
        neuro_data_spec={
            'start_time_before_event': -2,
            'end_time_after_event': 4,
            'bin_size': 0.2,
        }
        # run_decoding_exp(exp_folder, neuro_data_spec,
        #                  analyses=[
        #                      # 'decode_logit',
        #                      # 'decode_value',
        #                      # 'decode_chosen_value',
        #                      'value_decode_neuron',
        #                      'task_var_decode_neuron',
        #                      'task_var_value_decode_neuron',
        #                  ])
        compile_decoding_results(exp_folder, neuro_data_spec, extract_feature_func=extract_feature_func)


exp_folders = [
    'exp_seg_millerrat55_dataprop',
    # 'exp_seg_millerrat64_dataprop',
    # 'exp_seg_millerrat70_dataprop',
]
if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
                             additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )
    # compile_perf_for_exps(exp_folders, 'exp_seg_millerrat_dataprop',
    #                       additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
    #                       additional_cog_keys={'model_identifier_keys':['trainval_percent']},
    #                       additional_rnn_agg={'trainval_percent': ('trainval_percent','max')},
    #                       additional_cog_agg={'trainval_percent': ('trainval_percent','max')})
for exp_folder in exp_folders:
    if 'run_scores_for_each_data_proportion' in analyzing_pipeline:
        run_scores_exp(exp_folder)

def _stay_prob_analysis(choices, transitions, outcomes, trial_select, sample_choices=None):
    '''Analysis for stay probability plots using binary mask to select trials.'''
    if sample_choices is None:
        sample_choices = choices
    stay = sample_choices[1:] == choices[:-1]
    transitions, outcomes, trial_select = (transitions[:-1], outcomes[:-1], trial_select[:-1])
    transitions = transitions.astype(bool)
    outcomes = outcomes.astype(bool)
    trial_select = trial_select.astype(bool)
    stay_go_by_type = [stay[( outcomes &  transitions) & trial_select],  # A transition, rewarded.
                       stay[( outcomes & ~transitions) & trial_select],  # B transition, rewarded.
                       stay[(~outcomes &  transitions) & trial_select],  # A transition, not rewarded.
                       stay[(~outcomes & ~transitions) & trial_select]]  # B transition, not rewarded.
    n_trials_by_type = [len(s) for s in stay_go_by_type]
    n_stay_by_type =   [sum(s) for s in stay_go_by_type]
    return n_trials_by_type, n_stay_by_type

def nansem(x,dim = 0, ddof = 1):
    '''Standard error of the mean ignoring nans along dimension dim.'''
    return np.sqrt(np.nanvar(x,dim)/(np.sum(~np.isnan(x),dim) - ddof))

def stay_probability_analysis(behav_dt, ebars='SEM',
                              ylim=[0.,1], trial_mask=None, title=None):
    '''Stay probability analysis.'''
    assert ebars in [None, 'SEM', 'SD'], 'Invalid error bar specifier.'
    n_sessions = behav_dt.batch_size
    # n_sessions = len(sessions)
    print('Stay probability analysis for {} sessions.'.format(n_sessions))
    behav= behav_dt.behav
    GRU_scores = []
    MB_scores = []
    # unique but not sorted
    a = np.array(behav['sub_id'])
    _, idx = np.unique(a, return_index=True)
    unique_sub_ids = a[np.sort(idx)]

    for i in unique_sub_ids:
        for k,v in behav_dt.subject_remapping.items():
            if v == i:
                break
        # est_dim = joblib.load(ANA_SAVE_PATH / f'exp_seg_millerrat{k}' / 'rnn_final_perf_est_dim.pkl')
        # est_dim = est_dim[est_dim['less_than_former']==1].iloc[-1]

        rnn_type = 'GRU' #est_dim['rnn_type']
        hidden_dim = 2 #est_dim['hidden_dim']
        readout_FC = True #est_dim['readout_FC']
        summary = joblib.load(ANA_SAVE_PATH / f'exp_seg_millerrat{k[1:]}' / f'rnn_final_best_summary.pkl')
        model_path = summary[(summary['rnn_type']==rnn_type) & (summary['hidden_dim']==hidden_dim) & (summary['readout_FC']==readout_FC)
                             ].iloc[0]['model_path']
        scores = joblib.load(ANA_SAVE_PATH / model_path / 'total_scores.pkl')['scores']
        GRU_scores.extend(scores)
        print(model_path)

        summary = joblib.load(ANA_SAVE_PATH / f'exp_seg_millerrat{k[1:]}' / f'cog_final_best_summary.pkl')
        model_path = summary[(summary['cog_type']=='MB1')
                             ].iloc[0]['model_path']
        scores = joblib.load(ANA_SAVE_PATH / model_path / 'total_scores.pkl')['scores']
        MB_scores.extend(scores)
        print(model_path)

    from plotting_experiments.plotting import plot_start
    for dm in ['Data', 'GRU', 'Model-based'
               ]:
        fig, ax = plot_start(figsize=(0.7,1.5))
        np.random.seed(0)
        all_n_trials, all_n_stay = (np.zeros([n_sessions,4]), np.zeros([n_sessions,4]))
        for i in range(n_sessions):
            choices = behav['action'][i]
            stage2 = behav['stage2'][i]
            transitions = (choices == stage2) * 1
            outcomes = behav['reward'][i]
            if dm == 'Data':
                sample_choices = choices
            else:
                if dm == 'GRU':
                    score = GRU_scores[i]
                elif dm == 'Model-based':
                    score = MB_scores[i]
                    if score.shape[0] == 2:
                        score = score.T
                # print(score.shape[0]-1, len(choices),i, behav['sub_id'][i])
                assert score.shape[0]-1 == len(choices) or score.shape[0] == len(choices), (score.shape, len(choices),i, behav['sub_id'][i])
                #print(score)
                score = score - score.max(axis=1, keepdims=True)
                prob = np.exp(score) / np.exp(score).sum(axis=1, keepdims=True)
                sample_choices = np.zeros_like(choices)
                for j in range(len(choices)):
                    sample_choices[j] = np.random.choice(np.arange(2), p=prob[j])
            assert ((choices == stage2) * 1 == transitions).all(), 'Choice and stage2 do not match.'
            # trial_select = select_trials(selection, block_type=block_type)
            # trial_select = mask_first_n(trial_trans_state, n=20)
            trial_select = np.ones_like(choices).astype(bool)
            #Eval total trials and number of stay trial for A and B blocks.
            all_n_trials[i,:4] , all_n_stay[i,:4]  = _stay_prob_analysis(choices, transitions, outcomes, trial_select, sample_choices=sample_choices)
        if not ebars: # Don't calculate cross-animal error bars.
            mean_stay_probs = np.nanmean(all_n_stay / all_n_trials, 0)
            y_err  = np.zeros(4)
        else:
            sub_ids = np.array(behav['sub_id'])
            unique_sub_ids = np.unique(sub_ids)
            n_subjects = len(unique_sub_ids)
            per_subject_stay_probs = np.zeros([n_subjects,4])
            for i in unique_sub_ids:
                session_mask = sub_ids == i # True for sessions with correct animal ID.
                per_subject_stay_probs[i,:] = sum(all_n_stay[session_mask,:],0) / sum(all_n_trials[session_mask,:],0)
            mean_stay_probs = np.nanmean(per_subject_stay_probs, 0)
            if ebars == 'SEM':
                y_err = nansem(per_subject_stay_probs, 0)
            else:
                y_err = np.nanstd(per_subject_stay_probs, 0)
        plt.bar(np.arange(1,5), mean_stay_probs, yerr=y_err, label=dm, edgecolor='k', color='none')
        # plt.legend()
        plt.ylim(ylim)
        plt.yticks([0,0.5,1])
        plt.ylabel('Stay probability')
        plt.xlim(0.4,4.6)

        plt.xticks([1,2,3,4],['1\nC', '1\nR', '0\nC', '0\nR'])
        plt.title(dm)
        plt.savefig(FIG_PATH / 'exp_seg_millerrat' / f'stay_prob_{dm}.pdf', bbox_inches='tight')
        plt.close()
        plt.show()

if 'analyze_stay_prob_for_each_exp' in analyzing_pipeline:
    behav_data_spec = {'animal_name': ['m55','m64','m70','m71'], 'max_segment_length': 150,
                       'behav_format': 'tensor',
                       }
    dt = Dataset('MillerRat',
                 behav_data_spec=behav_data_spec).behav_to(behav_data_spec)
    stay_probability_analysis(dt, ylim=[0,1]
                              )#, by_trans=True)
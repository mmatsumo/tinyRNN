from plotting import *
from plotting_dynamics import *

save_pdf = True
plotting_pipeline = [
    'plot_model_perf_for_each_exp',
    # 'embedding_correlation'
]
exp_folders = [
    'exp_seg_akamrat49_distill_omni',
]

def plot_all_model_losses(exp_folder, xlim=None, ylim=None,  xticks=None, yticks=None,
                          max_hidden_dim=20, minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models',
                          model_curve_setting=None, add_text=False, save_pdf=True):

    if figsize is None:
        figsize = (1.5, 1.5)

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    fig, ax = plot_start(figsize=figsize)

    rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    # plot_figure = 'hidden_dim'
    plot_figure = 'trainval_size'
    # plot_figure = 'teacher_prop'
    # plot_figure = 'final_comparison'
    # with pd_full_print_context():
    #     print(rnn_perf)
    if plot_figure == 'hidden_dim':
        rnn_perf = rnn_perf[(rnn_perf['trainval_size'] == 107) & (rnn_perf['distill'] != 'none')]
        hidden_dims = pd.unique(rnn_perf['hidden_dim'])
        print(hidden_dims)

        for hidden_dim in hidden_dims:# [100,200,400]:
            this_rnn_perf = rnn_perf[rnn_perf['hidden_dim'] == hidden_dim]
            embedding_dim = this_rnn_perf['embedding_dim_c1']
            include_embedding = this_rnn_perf['include_embedding']
            embedding_dim *= include_embedding # set to 0 if not include_embedding
            perf = this_rnn_perf[perf_type]
            print(perf_type, hidden_dim,np.array(embedding_dim), np.array(perf))
            if len(perf) == 1:
                plt.scatter(embedding_dim, perf, label=f'GRU({hidden_dim})')
            else:
                plt.plot(embedding_dim, perf, label=f'GRU({hidden_dim})')
        plt.xlabel('# Embedding dimensions')
    elif plot_figure == 'trainval_size':
        this_rnn_perf = rnn_perf[(rnn_perf['hidden_dim'] == 4)]# & (rnn_perf['distill'] != 'none')]
        train_trial_num = this_rnn_perf['train_trial_num']
        # perf = this_rnn_perf[perf_type]
        # plt.plot(train_trial_num, perf, label=f'GRU({4})')

        rnn_perf = rnn_perf[rnn_perf['hidden_dim'] == 50]
        embedding_dims = pd.unique(rnn_perf['embedding_dim_c1'] * rnn_perf['include_embedding'])
        print(embedding_dims)
        for embedding_dim in embedding_dims[:4]:
            if embedding_dim == 0:
                this_rnn_perf = rnn_perf[rnn_perf['include_embedding'] == False]
            else:
                this_rnn_perf = rnn_perf[(rnn_perf['embedding_dim_c1'] == embedding_dim) & (rnn_perf['include_embedding'] == True)]
            trainval_size = this_rnn_perf['trainval_size']
            perf = this_rnn_perf[perf_type]
            # if perf_type == 'test_loss':
            #     print(train_trial_num, perf)
            plt.plot(trainval_size, perf, label=f'GRU(ebd{embedding_dim})')
        plt.ylim([0.575,0.7])
        # plt.xlabel('# training trials from this subject')
        # xticks = np.arange(0, 15000, 2000)
        # plt.xticks(xticks, rotation=90)
    elif plot_figure == 'teacher_prop':
        rnn_perf_teacher = rnn_perf[(rnn_perf['hidden_dim'] == 20) & (rnn_perf['embedding_dim'] == 8) & (rnn_perf['include_embedding'] == True)]
        rnn_perf_none = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'none')]
        rnn_perf_student = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'student')]
        train_trial_num = rnn_perf_none['train_trial_num']
        plt.plot(train_trial_num, rnn_perf_teacher[perf_type], label=f'Teacher GRU(20)')
        plt.plot(train_trial_num, rnn_perf_none[perf_type], label=f'Ori GRU(4)')
        for teacher_prop in pd.unique(rnn_perf_student['teacher_prop']):
            this_rnn_perf_student = rnn_perf_student[rnn_perf_student['teacher_prop'] == teacher_prop]
            plt.plot(train_trial_num, this_rnn_perf_student[perf_type], label=f'Student GRU(4) {teacher_prop}')
        plt.xlabel('# training trials from this subject')
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000], rotation=90)

    elif plot_figure == 'final_comparison':
        curve_alpha = 0.8
        dot_alpha = 0.6
        markersize = curve_markersize = 2.5
        MF_color = 'C4'
        GRU_color = 'C0'
        teacher_color =  'C7'
        student_color = 'C2'
        model_curves = {
            'solo': ModelCurve('solo GRU', 'solo GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
            'teacher': ModelCurve('teacher GRU', 'teacher GRU', teacher_color, curve_alpha, 'x', curve_markersize, 1, '-'),
            'student': ModelCurve('student GRU', 'student GRU', student_color, curve_alpha, 'x', curve_markersize, 1, '-'),
            'MF_dec_bs_rb_ck': ModelCurve('MF', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
        }

        labels_done = []

        cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
        teacher_perf = joblib.load(ana_exp_path / f'rnn_final_best_summary_this_animal_val_loss.pkl')
        cog_perf = cog_perf[cog_perf['trainval_size'] >=3]
        rnn_perf = rnn_perf[rnn_perf['trainval_size'] >=3]
        teacher_perf = teacher_perf[teacher_perf['trainval_size'] >=3]

        rnn_perf_teacher = teacher_perf[(teacher_perf['hidden_dim'] == 20)]
        rnn_perf_none = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'none')]
        rnn_perf_student = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'student')]
        trainval_size = rnn_perf_teacher['trainval_size']
        train_trial_num = rnn_perf_none['train_trial_num']
        print(np.array(trainval_size), np.array(train_trial_num))
        plot_trial_num_perf(train_trial_num, rnn_perf_none[perf_type], model_curves['solo'], labels_done)
        plot_trial_num_perf(train_trial_num, rnn_perf_teacher[perf_type], model_curves['teacher'], labels_done)
        plot_trial_num_perf(train_trial_num, rnn_perf_student[perf_type], model_curves['student'], labels_done)

        cog_models = pd.unique(cog_perf['cog_type'])
        assert len(cog_models) == 1 # for now
        for cog_type in cog_models:
            this_cog_perf = cog_perf[cog_perf['cog_type'] == cog_type]
            train_trial_num = this_cog_perf['train_trial_num']
            #print(this_cog_perf['test_loss'])
            labels_done = plot_trial_num_perf(train_trial_num, this_cog_perf['test_loss'], model_curves[cog_type], labels_done)

        plt.xlabel('# Trials for training')
        plt.xticks([0, 6000, 12000])

    plt.ylabel(f'Negative log likelihood')
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('Hidden units')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    if add_text:
        figname = figname + '_text'
    figname = f'{figname}_{plot_figure}_{perf_type}' + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()

if 'plot_model_perf_for_each_exp' in plotting_pipeline:
    plot_all_model_losses(exp_folders[0], perf_type='test_loss')
    plot_all_model_losses(exp_folders[0], perf_type='train_loss')
    plot_all_model_losses(exp_folders[0], perf_type='val_loss')

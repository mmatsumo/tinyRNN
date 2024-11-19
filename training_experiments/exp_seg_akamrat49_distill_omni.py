"""
Run teacher networks on Akam's rat 49.
Especially the blocks are segmented.
Vary the data proportion.
Using the Omni dataset.
"""
import sys
from sklearn.model_selection import train_test_split
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

if __name__ == '__main__' or '.' in __name__:
    # when animal_name is 'all', max_segment_length is 150, then [:143] belongs to Akam rat 49. [143:3757] belongs to others
    # split training and testing data for range(0, 143)
    # this animal and other animals
    max_idx = 3757
    this_idx = list(range(0, 143))
    other_idx = list(range(143, max_idx))
    other_trainval_idx = other_idx
    other_train_idx, other_val_idx = train_test_split(other_trainval_idx, test_size=0.1, random_state=0) # previous test_size=0.2
    this_trainval_idx, this_test_idx = train_test_split(this_idx, test_size=0.25, random_state=0) # previous test_size=0.5
    this_trainval_subset_size_list = [2,3,5,10,20,30,40,50,60,70,80,90,100,107]
    for this_trainval_subset_size in this_trainval_subset_size_list:
        np.random.seed(0)
        this_trainval_subset_idx = this_trainval_idx[:this_trainval_subset_size]
        this_val_subset_size = int(this_trainval_subset_size * 0.1)
        if this_val_subset_size == 0:
            this_val_subset_size = 1
        this_train_subset_idx, this_val_subset_idx = train_test_split(this_trainval_subset_idx, test_size=this_val_subset_size, random_state=0)
        train_idx = this_train_subset_idx + other_train_idx
        val_idx = this_val_subset_idx + other_val_idx
        test_idx = this_test_idx
        # print('this_trainval_subset_size', this_trainval_subset_size)
        # print('this_train_subset_idx', len(this_train_subset_idx))
        # print('this_val_subset_idx', len(this_val_subset_idx))
        # print('train_idx', len(train_idx))
        # print('val_idx', len(val_idx))
        # print('test_idx', len(test_idx))


        base_config = {
            ### dataset info
            'dataset': 'Omni',
            'behav_format': 'tensor',
            'behav_data_spec': ['animal_name', 'max_segment_length','include_embedding','include_task'],
            'animal_name': 'all',
            'max_segment_length': 150,
            ### model info
            'agent_type': 'RNN',
            'rnn_type': 'GRU', # which rnn layer to use

            'include_task': True, # should make sure include_embedding is True
            'include_embedding': True,  # if False, then the embedding layer is ignored (also affects include_task?)
            'num_embedding_channels': 2, # task embedding and animal embedding
            'num_embeddings_c0': 5,
            'embedding_dim_c0': 5,
            'embedding_init_c0': 'id',
            'num_embeddings_c1': 44,
            'embedding_dim_c1': 4,
            'embedding_init_c1': 'zero',

            'input_dim': 3,
            'hidden_dim': 2, # dimension of this rnn layer
            'output_dim': 2, # dimension of action
            'device': 'cuda',
            'output_h0': True, # whether initial hidden state included in loss
            'trainable_h0': False, # the agent's initial hidden state trainable or not
            'readout_FC': True, # whether the readout layer is full connected or not
            'one_hot': False, # whether the data input is one-hot or not
            ### training info for one model
            'batch_size': 1000,
            'lr':0.005,
            'l1_weight': 1e-5,
            'weight_decay': 0,
            'penalized_weight': 'rec',
            'max_epoch_num': 2000,
            'early_stop_counter': 200,
            ### training info for many models on dataset
            'split_training': True,
            'train_index': train_idx,
            'val_index': val_idx,
            'test_index': test_idx,
            'seed_num': 3,
            ### additional training info
            'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
            'training_diagnose': ['plot_loss'], # can be a list of diagnose function strings, e.g. ['plot_loss']
            ### current training exp path
            'exp_folder': get_training_exp_folder_name(__file__),
        }

        config_ranges = {  # keys are used to generate model names
            'rnn_type': ['GRU'],
            'hidden_dim': [
                # 20,
                50,
                #  100,
                # 200,
            ],
            'l1_weight': [1e-5],
            'include_embedding': [True],
            'embedding_dim_c1': [10,9,8,7,6, 5, 4, 3, 2, 1],
            'trainval_size': [this_trainval_subset_size],
            'distill': ['teacher'], # teacher will use all allowed data on all animals
        }

        resource_dict = {'memory': 15, 'cpu': 1, 'gpu': 1}
        # teacher
        behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
        # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)


        # # student
        # teacher_summary = joblib.load(ANA_SAVE_PATH / 'exp_seg_akamrat49_distill' / f'rnn_final_best_summary_this_animal_val_loss.pkl')
        # teacher_summary = teacher_summary[(teacher_summary['distill'] == 'teacher') & (teacher_summary['trainval_size'] == this_trainval_subset_size)]
        # assert len(teacher_summary) == 1
        # teacher_summary = teacher_summary.iloc[0]
        # teacher_model_path = teacher_summary['model_path']
        # dt = Dataset('AkamRat',
        #              behav_data_spec={'animal_name': 'all', 'max_segment_length': 150, 'include_embedding': True, 'augment': True,},verbose=False)
        # base_config.update({
        #     'behav_data_spec': ['animal_name', 'max_segment_length', 'include_embedding','augment'],
        #     'train_index': dt.get_after_augmented_block_number(this_train_subset_idx),
        #     'val_index': dt.get_after_augmented_block_number(this_val_subset_idx),
        #     'test_index': this_test_idx,
        #     'augment': True, # whether to augment the data
        #     'seed_num': 3,
        #     'teacher_model_path': teacher_model_path,
        # })
        # config_ranges.update({
        #     'hidden_dim': [4],
        #     'l1_weight': [1e-5,# 1e-4, 1e-3,
        #                   #1e-2, 1e-1
        #                   ],
        #     'include_embedding': [False],
        #     'distill': ['student'],
        # })
        # # behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1)


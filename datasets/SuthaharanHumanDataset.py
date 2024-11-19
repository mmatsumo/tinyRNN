import os
import joblib
import scipy.io
import numpy as np
import torch
from .BaseTwoStepDataset import BaseTwoStepDataset
import pandas as pd

class SuthaharanHumanDataset(BaseTwoStepDataset):
    """A dataset class for the 3-arm bandit task with 3 actions and binary rewards.

    Attributes:
         unique_trial_type: 3*2 = 6
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=True):
        self.unique_trial_type = 6
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def _detect_trial_type(self):
        """Determine trial type from behavioral data."""
        behav = self.behav
        behav['trial_type'] = []
        for b in range(self.batch_size):
            behav['trial_type'].append(behav['action'][b] * 2 + behav['reward'][b])

    def _load_all_trial_type(self, behav_data_spec):
        """Create artificial dataset with all trial types."""
        pass

    def get_behav_data(self, batch_indices, format_config=None, remove_padding_trials=False, selected_trial_indices=None):
        """ overwrite the get_behav_data function in BaseTwoStepDataset with remove_padding_trials option"""
        assert not remove_padding_trials
        return super().get_behav_data(batch_indices, format_config, remove_padding_trials, selected_trial_indices)

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load behavioral and neural data.

        The loaded data is stored in self.behav, the standard format of behavioral data.

        Args:
            behav_data_spec: A dictionary of behavioral data specification.
            neuro_data_spec: A dictionary of neural data specification.
                select_bins
         """

        if neuro_data_spec is None:
            neuro_data_spec = {}
        self.behav_data_spec = behav_data_spec
        self.neuro_data_spec = neuro_data_spec

        mat_files = [
            'pandemic_lockdown_TP463_TP693_231.mat',
            'pandemic_postlockdown_TP694_TP865_172.mat',
            'pandemic_prelockdown_TP117_TP261_130.mat',
            'replication_R100_R150_51.mat',
            'replication_R151_R198_48.mat',
            'replication_R199_R256_58.mat',
            'replication_R257_R357_101.mat',
            'replication_R358_R504_147.mat'
        ]

        self.behav = behav = {
            "sub_id": [],
            "action": [],
            "reward": []
        }
        self.subject_remapping = subject_remapping = {}
        sub_id = 0
        for mat_file in mat_files:
            mat_content = scipy.io.loadmat(self.data_path / mat_file)

            processed_ids = set()
            all_sub_ids = []
            all_choices = []
            all_outcomes = []

            for key in mat_content.keys():
                if "_1" in key:  # Use only the _1 variables to identify unique subjects
                    assert "_V3_" in key
                    sub_name = key.split("_V3_")[0]
                    if sub_name not in processed_ids:  # Ensure we haven't processed this subject ID yet
                        processed_ids.add(sub_name)
                        subject_remapping[sub_name] = sub_id
                        all_sub_ids.append(sub_id)
                        sub_id += 1

                        choices_1 = mat_content[f"{sub_name}_V3_choices_1"]
                        choices_2 = mat_content[f"{sub_name}_V3_choices_2"]
                        all_choices.append(np.concatenate([choices_1,choices_2]).flatten()-1)
                        assert np.unique(all_choices[-1]).size == 3, np.unique(all_choices[-1])

                        outcomes_1 = mat_content[f"{sub_name}_V3_outcomes_1"]
                        outcomes_2 = mat_content[f"{sub_name}_V3_outcomes_2"]
                        all_outcomes.append(np.concatenate([outcomes_1,outcomes_2]).flatten())
                        assert np.unique(all_outcomes[-1]).size == 2, np.unique(all_outcomes[-1])

                        # print(all_choices[-1].shape, all_outcomes[-1].shape)
                        # syss

            behav["sub_id"].extend(all_sub_ids)
            behav["action"].extend(all_choices)
            behav["reward"].extend(all_outcomes)
        behav['blockinfo']=[np.array([0]*39+[0.25]*40+[0.5]*40+[0.75]*40+[1])]*len(behav['sub_id'])

        behav['aug_block_number'] = list(np.arange(len(behav['action']))) # list of numbers, the block number to be augmented
        if 'augment' in behav_data_spec and behav_data_spec['augment']: # only used in distillation
            self._augment_data()
        self._detect_trial_type()
        print("===loaded all===", 'Suthaharan Human')
        print('Total batch size:', self.batch_size)
        print('Total trial num:', self.total_trial_num)

    def _behav_to_tensor(self, format_config):
        """Transform standard behavioral format to tensor format, stored in torch_beahv_* attribute.

        standard format (list of 1d array) -> tensor format (2d array with 0 padding).
        The attributes are:
            torch_beahv_input: tensor format of agent's input
            torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
            torch_beahv_target: tensor format of agent's target output
            torch_beahv_mask: tensor format of agent's mask (1 for valid trials, 0 for padding trials)

        Not use nan padding:
            rnn model make all-nan output randomly (unexpected behavior, not sure why)
            the one_hot function cannot accept nan
            long type is required for cross entropy loss, but does not support nan value

        Args:
            format_config: A dict specifies how the standard data should be transformed.

        """
        if self.torch_beahv_input is not None:
            return
        max_trial_num = max([len(block) for block in self.behav['reward']])
        assert max_trial_num == 160
        include_task = 'include_task' in format_config and format_config['include_task']
        include_embedding = 'include_embedding' in format_config and format_config['include_embedding']
        include_block = 'blockinfo' in format_config and format_config['blockinfo']
        self.include_embedding = include_embedding
        self.include_task = include_task
        self.include_block = include_block
        act = np.zeros((self.batch_size, max_trial_num, 3)).astype(np.int32)
        rew = np.zeros((self.batch_size, max_trial_num))
        target = np.zeros((self.batch_size, max_trial_num))
        task = np.zeros((self.batch_size, max_trial_num))
        sub = np.zeros((self.batch_size, max_trial_num))
        mask = np.ones((self.batch_size, max_trial_num))
        block = np.zeros((self.batch_size, max_trial_num))
        for b in range(self.batch_size):
            this_trial_num = len(self.behav['reward'][b])
            assert this_trial_num == 160
            act_temp = self.behav['action'][b].copy()
            act[b] = np.eye(3)[act_temp.astype(np.int32)]# onehot encoding of action
            rew[b] = self.behav['reward'][b]
            target[b] = self.behav['action'][b]
            if include_task:
                task[b, :] = self.behav['task_id'][b]
            if include_embedding:
                this_sub_id = np.unique(self.behav['sub_id'][b])
                assert len(this_sub_id) == 1
                sub[b, :] = this_sub_id
            if include_block:
                block[b, :] = self.behav['blockinfo'][b]
        device = 'cpu' if 'device' not in format_config else format_config['device']
        output_h0 = True if 'output_h0' not in format_config else format_config['output_h0']

        act = torch.from_numpy(np.swapaxes(act, 0,1)).to(device=device)  # act shape: trial_num, batch_size, 3
        rew = torch.from_numpy(rew.T[..., None]).to(device=device)
        target = torch.from_numpy(np.swapaxes(target, 0,1)).to(device=device) # target shape: trial_num, batch_size
        if output_h0:
            input = torch.cat([act, rew], -1)  # trial_num, batch_size, input_size=3
            target = target
            # print('output_h0', output_h0, 'h0 included in target')
        else:
            input = torch.cat([act, rew], -1)[:-1]
            target = target[1:]  # class, not one-hot
            # print('output_h0', output_h0, 'h0 excluded in target')

        if include_task:
            task = torch.from_numpy(np.swapaxes(task[..., None], 0,1)).to(device=device)
            input = torch.cat([input, task], -1)

        if include_block:
            block = torch.from_numpy(np.swapaxes(block[..., None], 0,1)).to(device=device)
            input = torch.cat([input, block], -1)
        if include_embedding:
            assert NotImplementedError
            sub = torch.from_numpy(np.swapaxes(sub[..., None], 0,1)).to(device=device)
            input = torch.cat([input, sub], -1) # input shape: trial_num, batch_size, 3+sub_num
        self.torch_beahv_input = input.double()
        self.torch_beahv_target = target.long()
        self.torch_beahv_mask = torch.from_numpy(mask.T).to(device=device).double()


    def _behav_to_cog_sessions(self, format_config):
        """Transform standard behavioral format to cog_session format, stored in cog_sessions attribute.

        Args:
            format_config: A dict specifies how the standard data should be transformed.
        """
        if self.cog_sessions is not None:
            return
        self.cog_sessions = []
        behav = self.behav
        print('Transforming standard format to cog_session format...')
        for block_idx in range(self.batch_size):
            action = behav['action'][block_idx].copy()
            reward = behav['reward'][block_idx].copy()
            trial_num = len(reward)
            self.cog_sessions.append({
                'n_trials': trial_num,
                'choices': action,
                'second_steps': action,
                'outcomes': reward,
                'mask': np.ones(trial_num).astype(np.int32),
            })
        print('\nTotal block num', self.batch_size)

    def _augment_data(self):
        """Augment data by rotating the action
        Only augment the data for the students. 3x"""
        behav = self.behav
        augment = self.behav_data_spec['augment']
        if augment is True or augment == 1:
            self.augment_ratio = 3
            perms = [
                [0, 1, 2],
                [1, 2, 0],
                [2, 0, 1],
            ]
        elif augment == 2:# second level
            self.augment_ratio = 6
            import itertools
            perms = list(itertools.permutations([0,1,2,]))
        else:
            raise NotImplementedError
        number_block_before_aug = len(behav['action'])
        def _flip(data, values1, values2):
            data_new = data.copy()
            for v1, v2 in zip(values1, values2):
                data_new[data == v1] = v2
            return data_new
        for rot in perms[1:]:# skip the first one (0,1,2,3)
            for b in range(number_block_before_aug):
                behav['action'].append(_flip(behav['action'][b], [0, 1, 2], rot))
                behav['reward'].append(behav['reward'][b])
                behav['sub_id'].append(behav['sub_id'][b])
                behav['blockinfo'].append(behav['blockinfo'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)


    def get_after_augmented_block_number(self, block_indices_before_augmentation):
        """Extract the block indices in augmented data, for each block-number-before-augmentation in block_indices_before_augmentation."""
        aug_block_number = np.array(self.behav['aug_block_number'])
        block_indices_after_augmentation = []
        for b in block_indices_before_augmentation: # for each block in the real data
            aug_idx = np.where(aug_block_number == b)[0]
            block_indices_after_augmentation.append(aug_idx)
        block_indices_after_augmentation = list(np.concatenate(block_indices_after_augmentation))
        assert len(block_indices_after_augmentation) == len(block_indices_before_augmentation) * self.augment_ratio
        return [int(x) for x in block_indices_after_augmentation] # convert to int, numpy.int64 is not json serializable
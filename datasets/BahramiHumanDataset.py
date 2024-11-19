import os
import joblib
import numpy as np
import torch
from .BaseTwoStepDataset import BaseTwoStepDataset
import pandas as pd

class BahramiHumanDataset(BaseTwoStepDataset):
    """A dataset class for the 4-arm bandit task with 4 actions and continuous rewards.

    Attributes:
         unique_trial_type: -1
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=False):
        self.unique_trial_type = -1
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def _detect_trial_type(self):
        """Determine trial type from behavioral data."""
        behav = self.behav
        behav['trial_type'] = []
        for b in range(self.batch_size):
            act_temp = self.behav['action'][b].copy().astype(np.int32)
            act_temp[np.isnan(act_temp)] = 0
            rew_temp = self.behav['reward'][b].copy() / self.reward_scale
            rew_temp[np.isnan(rew_temp)] = 0
            trial_type = list(zip(act_temp, rew_temp))
            behav['trial_type'].append(trial_type)

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

        data = pd.read_csv(self.data_path / '4ArmBandit_DataAllSubjectsRewards.csv')

        # Initialize the result dictionary
        self.behav = behav = {'sub_id': []}
        self.subject_remapping = subject_remapping = {}
        sub_id = 0
        skip_sub = []
        for name, group in data.groupby('id'):
            nans = np.sum(np.isnan(group['choice'].values))
            if nans > 15:
                if verbose: print('skip sub', name, 'nans',nans)
                skip_sub.append(name)
                continue
            subject_remapping[name] = sub_id
            behav['sub_id'].append(sub_id)
            sub_id += 1
            for column in group.columns:
                if column != 'id':
                    behav.setdefault(column, []).append(group[column].values)
            behav.setdefault('action', []).append(group['choice'].values - 1)
        behav.pop('choice')
        if verbose: print('skip sub', skip_sub, 'total', len(skip_sub))

        behav['aug_block_number'] = list(np.arange(len(behav['action']))) # list of numbers, the block number to be augmented
        if 'augment' in behav_data_spec and behav_data_spec['augment']: # only used in distillation
            self._augment_data()
        self.reward_scale = 100
        self._detect_trial_type()
        print("===loaded all===", 'Bahrami Human')
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
        assert max_trial_num == 150
        include_task = 'include_task' in format_config and format_config['include_task']
        include_embedding = 'include_embedding' in format_config and format_config['include_embedding']
        self.include_embedding = include_embedding
        self.include_task = include_task
        act = np.zeros((self.batch_size, max_trial_num, 4))
        rew = np.zeros((self.batch_size, max_trial_num))
        target = np.zeros((self.batch_size, max_trial_num))
        task = np.zeros((self.batch_size, max_trial_num))
        sub = np.zeros((self.batch_size, max_trial_num))
        mask = np.ones((self.batch_size, max_trial_num))
        for b in range(self.batch_size):
            this_trial_num = len(self.behav['reward'][b])
            assert this_trial_num == 150
            act_temp = self.behav['action'][b].copy()
            act_temp[np.isnan(act_temp)] = 0
            act[b] = np.eye(4)[act_temp.astype(np.int32)]# onehot encoding of action
            mask[b, np.isnan(self.behav['action'][b])] = 0

            reward_temp = self.behav['reward'][b].copy()
            reward_temp[np.isnan(reward_temp)] = 0
            mask[b, np.isnan(self.behav['reward'][b])] = 0
            rew[b] = reward_temp / self.reward_scale
            target[b] = act_temp
            if include_task:
                task[b, :] = self.behav['task_id'][b]
            if include_embedding:
                this_sub_id = np.unique(self.behav['sub_id'][b])
                assert len(this_sub_id) == 1
                sub[b, :] = this_sub_id
        device = 'cpu' if 'device' not in format_config else format_config['device']
        output_h0 = True if 'output_h0' not in format_config else format_config['output_h0']

        act = torch.from_numpy(np.swapaxes(act, 0,1)).to(device=device)  # act shape: trial_num, batch_size, 4
        rew = torch.from_numpy(rew.T[..., None]).to(device=device)
        target = torch.from_numpy(target.T).to(device=device) # target shape: trial_num, batch_size
        if output_h0:
            input = torch.cat([act, rew], -1)  # trial_num, batch_size, input_size=5
            target = target  # class, not one-hot
            # print('output_h0', output_h0, 'h0 included in target')
        else:
            input = torch.cat([act, rew], -1)[:-1]
            target = target[1:]  # class, not one-hot
            # print('output_h0', output_h0, 'h0 excluded in target')

        if include_task:
            task = torch.from_numpy(np.swapaxes(task[..., None], 0,1)).to(device=device)
            input = torch.cat([input, task], -1)

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
            mask = np.ones_like(action)
            mask[np.isnan(action)] = 0
            action[np.isnan(action)] = 0
            reward = behav['reward'][block_idx].copy()
            mask[np.isnan(reward)] = 0
            reward[np.isnan(reward)] = 0
            reward /= self.reward_scale
            trial_num = len(reward)
            self.cog_sessions.append({
                'n_trials': trial_num,
                'choices': action.astype(np.int32),
                'second_steps': action.astype(np.int32),
                'outcomes': reward,
                'mask': mask.astype(np.int32),
            })
        print('\nTotal block num', self.batch_size)

    def _augment_data(self):
        """Augment data by rotating the action
        Only augment the data for the students. 4x"""
        behav = self.behav
        augment = self.behav_data_spec['augment']
        if augment is True or augment == 1:
            self.augment_ratio = 4
            perms = [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
                [2, 3, 0, 1],
                [3, 0, 1, 2],
            ]
        elif augment == 2:# second level
            self.augment_ratio = 24
            import itertools
            perms = list(itertools.permutations([0,1,2,3]))
        else:
            raise NotImplementedError
        number_block_before_aug = len(behav['action'])
        def _flip(data, values1, values2):
            data_new = data.copy()
            for v1, v2 in zip(values1, values2):
                data_new[data == v1] = v2
            return data_new
        for rot in perms[1:]: # skip the first one (0,1,2,3)
            for b in range(number_block_before_aug):
                behav['action'].append(_flip(behav['action'][b], [0, 1, 2, 3], rot))
                behav['reward'].append(behav['reward'][b])
                behav['sub_id'].append(behav['sub_id'][b])
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
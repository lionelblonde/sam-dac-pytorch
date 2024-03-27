from collections import defaultdict

import numpy as np

from helpers.math_util import discount


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class RingBuffer(object):

    def __init__(self, maxlen, shape, dtype='float32'):
        # ring buffer implementation
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen, *shape), dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # we have space, simply increase the length
            self.length += 1
            self.data[(self.start + self.length - 1) % self.maxlen] = v

        elif self.length == self.maxlen:
            # no space, remove the first item
            self.start = (self.start + 1) % self.maxlen
            self.data[(self.start + self.length - 1) % self.maxlen] = v

        else:
            # this should never happen
            raise RuntimeError()


class ReplayBuffer(object):

    def __init__(self, np_rng, capacity, shapes):
        self.np_rng = np_rng
        self.capacity = capacity
        self.shapes = shapes
        self.ring_buffers = {n: RingBuffer(self.capacity, s) for n, s in self.shapes.items()}

    def batchify(self, idxs):
        # collect a batch from indices
        transitions = {
            n: array_min2d(self.ring_buffers[n].get_batch(idxs))
            for n in self.ring_buffers.keys()
        }
        transitions['idxs'] = idxs  # add idxs too
        return transitions

    def sample(self, batch_size, patcher):
        # sample transitions uniformly from the replay buffer
        idxs = self.np_rng.integers(low=0, high=self.num_entries, size=batch_size)
        transitions = self.batchify(idxs)

        if patcher is not None:
            # patch the rewards
            transitions['rews'] = patcher(
                transitions['obs0'],
                transitions['acs'],
                transitions['obs1'],
            )

        return transitions

    def lookahead(self, transitions, n, gamma, patcher):
        # perform n-step TD lookahead estimations starting from every transition"""
        assert 0 <= gamma <= 1

        # initiate the batch of transition data necessary to perform n-step TD backups
        la_batch = defaultdict(list)

        # iterate over the indices to deploy the n-step backup for each
        for idx in transitions['idxs']:
            # create indexes of transitions in lookahead of lengths max `n` following sampled one
            la_end_idx = min(idx + n, self.num_entries) - 1
            la_idxs = np.array(range(idx, la_end_idx + 1))
            # collect the batch for the lookahead rollout indices
            la_transitions = self.batchify(la_idxs)
            if patcher is not None:
                # patch the rewards
                la_transitions['rews'] = patcher(
                    la_transitions['obs0'],
                    la_transitions['acs'],
                    la_transitions['obs1'],
                )
            # only keep data from the current episode, drop everything after episode reset, if any
            dones = la_transitions['dones1']
            ep_end_idx = idx + list(dones).index(1.0) if 1.0 in dones else la_end_idx
            la_is_trimmed = 0.0 if ep_end_idx == la_end_idx else 1.0
            # compute lookahead length
            td_len = ep_end_idx - idx + 1
            # trim down the lookahead transitions
            la_rews = la_transitions['rews'][:td_len]
            # compute discounted cumulative reward
            la_discounted_sum_n_rews = discount(la_rews, gamma)[0]
            # populate the batch for this n-step TD backup
            la_batch['obs0'].append(la_transitions['obs0'][0])
            la_batch['obs1'].append(la_transitions['obs1'][td_len - 1])
            la_batch['acs'].append(la_transitions['acs'][0])
            la_batch['rews'].append(la_discounted_sum_n_rews)
            la_batch['dones1'].append(la_is_trimmed)
            la_batch['td_len'].append(td_len)

            # add the first next state too: needed in state-only discriminator
            la_batch['obs1_td1'].append(la_transitions['obs1'][0])

            # when dealing with absorbing states
            if 'obs0_orig' in la_transitions.keys():
                la_batch['obs0_orig'].append(la_transitions['obs0_orig'][0])
            if 'obs1_orig' in la_transitions.keys():
                la_batch['obs1_orig'].append(la_transitions['obs1_orig'][td_len - 1])
            if 'acs_orig' in la_transitions.keys():
                la_batch['acs_orig'].append(la_transitions['acs_orig'][0])

        la_batch['idxs'] = transitions['idxs']

        # wrap every value with `array_min2d`
        la_batch = {k: array_min2d(v) for k, v in la_batch.items()}
        return la_batch

    def lookahead_sample(self, batch_size, n, gamma, patcher):
        # sample a batch of transitions
        transitions = self.sample(batch_size, patcher)
        # expand each transition with a n-step TD lookahead
        return self.lookahead(transitions, n, gamma, patcher)

    def append(self, transition):
        # add a transition to the replay buffer
        assert self.ring_buffers.keys() == transition.keys(), "keys must coincide"
        for k in self.ring_buffers.keys():
            self.ring_buffers[k].append(transition[k])

    def __repr__(self):
        return "ReplayBuffer(capacity={})".format(self.capacity)

    @property
    def latest_entry_idx(self):
        pick = self.ring_buffers['obs0']  # could pick any other key
        return (pick.start + pick.length - 1) % pick.maxlen

    @property
    def num_entries(self):
        return len(self.ring_buffers['obs0'])  # could pick any other key


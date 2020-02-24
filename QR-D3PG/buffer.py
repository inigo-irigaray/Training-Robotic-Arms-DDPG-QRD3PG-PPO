import torch
import operator
import random

import numpy as np

from collections import deque
from torch.autograd import Variable




class ReplayBuffer:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obsmemory = []
        self.actmemory = []
        self.rmemory = []
        self.nobsmemory = []
        self.donememory = []
        self.filled = 0
        self.current = 0
        
    def __len__(self):
        return self.filled
    
    def add(self, observations, actions, rewards, nobservations, dones):
        for i in range(self.num_agents):
            if self.filled < self.capacity:
                self.filled += 1
                self.current += 1
                self.obsmemory.append(observations[i,:])
                self.actmemory.append(actions[i,:])
                self.rmemory.append(rewards[i])
                self.nobsmemory.append(nobservations[i,:])
                self.donememory.append(dones[i])
            
            else: # when capacity is overblown, the buffer restarts storage from position 0
                self.obsmemory[self.current % self.capacity] = observations[i,:]
                self.actmemory[self.current % self.capacity] = actions[i,:]
                self.rmemory[self.current % self.capacity] = rewards[i]
                self.nobsmemory[self.current % self.capacity] = nobservations[i,:]
                self.donememory[self.current % self.capacity] = dones[i]
                self.current += 1
        
    def sample(self, batch_size=1, to_gpu=False, norm_r=False):
        """ Random sampling experiences"""
        idxs = np.random.choice(np.arange(self.filled), size=batch_size, replace=False)
        device = 'cpu'
        if to_gpu:
            device = 'cuda'
        # normalizes rewards automatically
        if norm_r: # TERRIBLE IDEA! SLOWS DOWN LEARNING DRAMATICALLY W/O SIGNIFICANT REWARD BOOST
            rewards = torch.from_numpy(np.vstack([
                (self.rmemory[idx] - np.mean(self.rmemory[:self.filled])) /
                np.std(self.rmemory[:self.filled]) for idx in idxs])).float().to(device)
        else:
            rewards = torch.from_numpy(np.vstack([self.rmemory[idx] for idx in idxs])).float().to(device)
            
        return (torch.from_numpy(np.vstack([self.obsmemory[idx] for idx in idxs])).float().to(device),
                torch.from_numpy(np.vstack([self.actmemory[idx] for idx in idxs])).float().to(device),
                rewards,
                torch.from_numpy(np.vstack([self.nobsmemory[idx] for idx in idxs])).float().to(device),
                torch.from_numpy(np.vstack([self.donememory[idx] for idx in idxs]).astype(np.uint8)).to(device))
    
    def get_average_rewards(self, N):
        assert self.filled > N # checks that the agent has played the minimum required games for the task
        if (self.current % self.capacity) < N:
            rewards_N = []
            rewards_N.append(self.rmemory[self.capacity-N-1:])
            rewards_N.append(self.rmemory[:self.current % self.capacity])
            return rewards_N.mean()
        return self.rmemory[self.capacity-N-1:].mean()
    
    
    
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, num_agents, alpha):
        super(PrioritizedReplayBuffer, self).__init__(capacity, num_agents)
        assert alpha > 0
        self._alpha = alpha
        
        it_capacity = 1
        while it_capacity < self.capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
    
    def add(self, *args, **kwargs):
        idx = self.current % self.capacity ### maybe move this one after the super() is called?
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
        
    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
        
    def sample(self, beta, batch_size=1, to_gpu=False, norm_r=False):
        assert beta > 0
        idxs = self._sample_proportional(batch_size)
        device = 'cpu'
        if to_gpu:
            device = 'cuda'

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxs:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
            
        weights = np.array(weights, dtype=np.float32)
        # normalizes rewards automatically
        if norm_r:
            rewards = torch.from_numpy(np.vstack([
                (self.rmemory[idx] - np.mean(self.rmemory[:self.filled])) /
                np.std(self.rmemory[:self.filled]) for idx in idxs])).float().to(device)
        else:
            rewards = torch.from_numpy(np.vstack([self.rmemory[idx] for idx in idxs])).float().to(device)
            
        return (torch.from_numpy(np.vstack([self.obsmemory[idx] for idx in idxs])).float().to(device),
                torch.from_numpy(np.vstack([self.actmemory[idx] for idx in idxs])).float().to(device),
                rewards,
                torch.from_numpy(np.vstack([self.nobsmemory[idx] for idx in idxs])).float().to(device),
                torch.from_numpy(np.vstack([self.donememory[idx] for idx in idxs]).astype(np.uint8)).to(device),
                idxs,
                weights)
        
    def update_priorities(self, idxs, prios):
        assert len(idxs) == len(prios)
        for idx, priority in zip(idxs, prios):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
    
    
    
    
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

    
    

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

    
    

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)    
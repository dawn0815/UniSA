import math
import torch
from torch.utils.data.sampler import RandomSampler
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets) 
        
        self.max_dataset_size=max([len(cur_dataset) for cur_dataset in dataset.datasets])
        
        # self.samples_rate=[ self.batch_size*l//sum(self.dataset_size) for l in self.dataset_size]
        # if sum(self.samples_rate)<self.batch_size:
        #     self.samples_rate[0]+=self.batch_size-sum(self.samples_rate)

    def __len__(self):
        return self.batch_size * (self.number_of_datasets*self.max_dataset_size // self.batch_size)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset) # 先对每个数据集的iterator进行shuffle
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] # 找到每个dataset第一个数据的index
        print('number of datasets(tasks): ',self.number_of_datasets)
        final_samples_list = []  # this is a list of indexes from the combined dataset
        steps=self.number_of_datasets*self.max_dataset_size//self.batch_size
        sample_num=[self.batch_size//self.number_of_datasets for _ in range(self.number_of_datasets)]
        if sample_num[0]*self.number_of_datasets<self.batch_size:
            sample_num[0]+=self.batch_size-sample_num[0]*self.number_of_datasets
        for _ in range(steps):
            cur_samples = []
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i] 
                for _ in range(sample_num[i]):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


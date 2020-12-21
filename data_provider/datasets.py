import numpy as np
import random

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.seq_length = input_param['sequence_length']
        self.joint_dim = input_param['joint_dim']
        self.joint_num = input_param['joint_num']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.load()

    def load(self):
        self.data = np.load(self.paths)
        print(self.data.shape)

    def total(self):
        return len(self.data)

    def begin(self, do_shuffle = True):
        self.indices = np.arange(self.total(),dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def input_batch(self):
        if self.no_batch_left():
            return None

        input_batch = np.zeros(
            (self.current_batch_size, self.seq_length, self.joint_num, self.joint_dim, 1)).astype(self.input_data_type)

        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]

            data_slice = self.data[batch_ind]
            data_slice = data_slice[0:self.seq_length]
            data_slice1=np.zeros([self.seq_length, self.joint_num, self.joint_dim, 1])
            data_slice1[:,:,:,0]=data_slice

            input_batch[i, :self.seq_length, :, :] = data_slice1
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def get_batch(self):
        batch = self.input_batch()
        return batch

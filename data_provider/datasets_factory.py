from data_provider import datasets

def data_provider(train_data_paths, valid_data_paths, batch_size,
                  joint_dim, joint_num, sequence_length, is_training=True):

    print('testing data :')
    test_input_param = {'paths': valid_data_paths,
                        'minibatch_size': batch_size,
                        'joint_dim':joint_dim,
                        'joint_num':joint_num,
                        'input_data_type': 'float32',
                        'sequence_length':sequence_length
                        }
    test_input_handle = datasets.InputHandle(test_input_param)
    test_input_handle.begin(do_shuffle = False)
    
    if is_training:
        print('training data :')
        train_input_param = {'paths': train_data_paths,
                             'minibatch_size': batch_size,
                             'joint_dim':joint_dim,
                             'joint_num':joint_num,
                             'input_data_type': 'float32',
                             'sequence_length':sequence_length
                             }
        train_input_handle = datasets.InputHandle(train_input_param)
        train_input_handle.begin(do_shuffle = True)

        return train_input_handle, test_input_handle
    else:
        return test_input_handle

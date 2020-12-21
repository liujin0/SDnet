import os.path
import numpy as np
import tensorflow as tf
from nets import static_dynamic_net
from data_provider import datasets_factory
from utils import optimizer
import time

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/g3d_dataset/G3dTrainData_no_norm_joints.npy',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/g3d_dataset/G3dTestData_no_norm_joints.npy',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints/g3d',
                            'dir to store trained net.')
# model
tf.app.flags.DEFINE_integer('max_saved_model', 10, 
							'maximum number of models saved')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 10,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 20,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('joint_dim', 3,
                            'dimensions of one joint.')
tf.app.flags.DEFINE_integer('joint_num', 18,
                            'the number of joints.')
tf.app.flags.DEFINE_integer('filter_size', 3,
                            'filter of a cascade multiplicative unit.')
tf.app.flags.DEFINE_string('num_hidden', '64',
                           'number of units in a cascade multiplicative unit.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'view skeleton as image; here 1 is the need of using tensorflow.')

# optimization
tf.app.flags.DEFINE_float('lr', 0.0001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 100000,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 10,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 500,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 500,
                            'number of iters saving models.')
tf.app.flags.DEFINE_integer('n_gpu', 1,
                            'how many GPUs to distribute the training across.')

# extra parameters of encoder and decoder
tf.app.flags.DEFINE_integer('encoder_length', 2,
                            'number of encoder residual multiplicative block of predCNN')
tf.app.flags.DEFINE_integer('decoder_length', 3,
                            'number of decoder residual multiplicative block of predCNN')


class Model(object):
    def __init__(self):
        # inputs
        self.x = [tf.placeholder(tf.float32,
                                 [FLAGS.batch_size,
                                  FLAGS.seq_length,
                                  FLAGS.joint_num,
                                  FLAGS.joint_dim,
                                  FLAGS.img_channel])
                  for i in range(FLAGS.n_gpu)]
                  
        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])

        self.params = dict()
        self.params['encoder_length'] = FLAGS.encoder_length
        self.params['decoder_length'] = FLAGS.decoder_length

        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        for i in range(FLAGS.n_gpu):
            with tf.device('/gpu:%d' % i):
                with tf.variable_scope(tf.get_variable_scope(),
                                       reuse=True if i > 0 else None):
                    # define a model
                    output_list = static_dynamic_net.static_dynamic_net(
                        self.x[i],
                        self.params,
                        num_hidden,
                        FLAGS.filter_size,
                        FLAGS.seq_length,
                        FLAGS.input_length)

                    gen_ims = output_list[0]
                    loss = output_list[1]
                    pred_ims = gen_ims[:, FLAGS.input_length - FLAGS.seq_length:]
                    loss_train.append(loss / FLAGS.batch_size)
                    # gradients
                    all_params = tf.trainable_variables()
                    grads.append(tf.gradients(loss, all_params))
                    self.pred_seq.append(pred_ims)

        if FLAGS.n_gpu == 1:
            self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
        else:
            # add losses and gradients together and get training updates
            with tf.device('/gpu:0'):
                for i in range(1, FLAGS.n_gpu):
                    loss_train[0] += loss_train[i]
                    for j in range(len(grads[0])):
                        grads[0][j] += grads[i][j]
            # keep track of moving average
            ema = tf.train.ExponentialMovingAverage(decay=0.9995)
            maintain_averages_op = tf.group(ema.apply(all_params))
            self.train_op = tf.group(optimizer.adam_updates(
                all_params, grads[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995),
                maintain_averages_op)

        self.loss_train = loss_train[0] / FLAGS.n_gpu

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables, max_to_keep=FLAGS.max_saved_model)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        feed_dict.update({self.tf_lr: lr})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs):
        feed_dict = {self.x[i]: inputs[i] for i in range(FLAGS.n_gpu)}
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


def main(argv=None):

    if ~tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.MakeDirs(FLAGS.save_dir)

    print('start training !',time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time())))
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(
        FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size * FLAGS.n_gpu,
        FLAGS.joint_dim,
        FLAGS.joint_num,
        FLAGS.seq_length,
        is_training=True)

    print('Initializing models')
    model = Model()
    lr = FLAGS.lr
    train_time=0
    test_time_all=0

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        start_time = time.time()
        ims = train_input_handle.get_batch()
        ims_list = np.split(ims, FLAGS.n_gpu)        
        cost = model.train(ims_list, lr)

        if FLAGS.reverse_input:
            ims_rev = np.split(ims[:, ::-1], FLAGS.n_gpu)
            cost += model.train(ims_rev, lr)
            cost = cost/2
        end_time = time.time()
        t=end_time-start_time
        train_time += t

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr)+' lr: '+str(lr)+' training loss: ' + str(cost))

        if itr % FLAGS.test_interval == 0:
            test_time=0

            print('train time:'+ str(train_time))
            print('test...')
            test_input_handle.begin(do_shuffle=False)
            
            mse_per_frame = []
            mae_per_frame = []
            while(test_input_handle.no_batch_left() == False):
                curr_test_time_start = time.time()
                test_ims = test_input_handle.get_batch()
                test_dat = np.split(test_ims, FLAGS.n_gpu)
                img_gen = model.test(test_dat)

                curr_test_time = time.time() - curr_test_time_start
                test_time += curr_test_time

                # concat outputs of different gpus along batch
                img_gen = np.concatenate(img_gen)
                absoult_err = np.squeeze(np.abs(test_ims[:, FLAGS.input_length:] - img_gen))
                absoult_err_t = np.transpose(absoult_err, [1, 0, 2, 3]).reshape([(FLAGS.seq_length-FLAGS.input_length), -1])
                mae = np.mean(absoult_err_t, axis=-1, keepdims=True)*FLAGS.joint_num*FLAGS.joint_dim
                mse = np.mean(absoult_err_t**2, axis=-1, keepdims=True)*FLAGS.joint_num*FLAGS.joint_dim
                mae_per_frame.append(mae)
                mse_per_frame.append(mse)

                test_input_handle.next()

            test_time_all += test_time
            print('current test time:'+str(test_time))
            print('all test time: '+str(test_time_all))
            mae_per_frame = np.concatenate(mae_per_frame).reshape([-1, (FLAGS.seq_length-FLAGS.input_length)]).mean(axis=0)
            mse_per_frame = np.concatenate(mse_per_frame).reshape([-1, (FLAGS.seq_length-FLAGS.input_length)]).mean(axis=0)

            print('average mse per frame: ', mse_per_frame.mean())
            _ = list(map(print, mse_per_frame))
            print('average mae per frame: ', mae_per_frame.mean())
            _ = list(map(print, mae_per_frame))

        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)
            print('model saving done! ', time.strftime('%Y-%m-%d %H:%M:%S\n\n\n',time.localtime(time.time())))

        train_input_handle.next()

if __name__ == '__main__':
    tf.app.run()

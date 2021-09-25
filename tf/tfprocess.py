#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import tensorflow as tf
import time
import bisect
import lc0_az_policy_map
import attention_policy_map as apm
import proto.net_pb2 as pb
from functools import reduce
import operator

from net import Net


class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
                                            [-1, self.reshape_size, 1, 1]),
                                 2,
                                 axis=1)
        return tf.nn.sigmoid(gammas) * x + betas


class ApplyPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(lc0_az_policy_map.make_map())

    def call(self, inputs):
        h_conv_pol_flat = tf.reshape(inputs, [-1, 80 * 8 * 8])
        return tf.matmul(h_conv_pol_flat,
                         tf.cast(self.fc1, h_conv_pol_flat.dtype))


class ApplyAttentionPolicyMap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplyAttentionPolicyMap, self).__init__(**kwargs)
        self.fc1 = tf.constant(apm.make_map())

    def call(self, logits):
        logits = tf.reshape(logits, [-1, 64 * 64])  # 64 * 88 <- for pawn promotion concept
        legal_logits = tf.matmul(logits, tf.cast(self.fc1, logits.dtype))
        # comment the next two lines, and un-comment the third for pawn promotion concept
        temp_promotions = tf.zeros([tf.shape(legal_logits)[0], 66], dtype=logits.dtype)  # <- set promotion logits to 0
        return tf.concat([legal_logits, temp_promotions], axis=1)
        # return legal_logits


class TFProcess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.net = Net()
        self.root_dir = os.path.join(self.cfg['training']['path'],
                                     self.cfg['name'])

        # Network structure
        self.RESIDUAL_FILTERS = self.cfg['model']['filters']
        self.RESIDUAL_BLOCKS = self.cfg['model']['residual_blocks']
        self.SE_ratio = self.cfg['model']['se_ratio']
        self.policy_channels = self.cfg['model'].get('policy_channels', 32)
        #########
        self.emb_size_pol = self.cfg['model'].get('emb_size_pol', 128)
        self.enc_layers_pol = self.cfg['model'].get('enc_layers_pol', 0)
        self.dff_pol_enc = self.cfg['model'].get('dff_pol_enc', 512)
        self.d_model_pol_enc = self.cfg['model'].get('d_model_pol_enc', 256)
        self.n_heads_pol_enc = self.cfg['model'].get('n_heads_pol_enc', 2)
        self.d_model_pol_hd = self.cfg['model'].get('d_model_pol_hd', 256)
        self.n_heads_pol_hd = self.cfg['model'].get('n_heads_pol_hd', 1)
        #########
        precision = self.cfg['training'].get('precision', 'single')
        loss_scale = self.cfg['training'].get('loss_scale', 128)
        self.virtual_batch_size = self.cfg['model'].get(
            'virtual_batch_size', None)

        if precision == 'single':
            self.model_dtype = tf.float32
        elif precision == 'half':
            self.model_dtype = tf.float16
        else:
            raise ValueError("Unknown precision: {}".format(precision))

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else loss_scale

        policy_head = self.cfg['model'].get('policy', 'convolution')
        value_head = self.cfg['model'].get('value', 'wdl')
        moves_left_head = self.cfg['model'].get('moves_left', 'v1')
        input_mode = self.cfg['model'].get('input_type', 'classic')

        self.POLICY_HEAD = None
        self.VALUE_HEAD = None
        self.MOVES_LEFT_HEAD = None
        self.INPUT_MODE = None

        if policy_head == "classical":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CLASSICAL
        elif policy_head == "convolution":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_CONVOLUTION
        elif policy_head == "attention":
            self.POLICY_HEAD = pb.NetworkFormat.POLICY_ATTENTION
        else:
            raise ValueError(
                "Unknown policy head format: {}".format(policy_head))

        self.net.set_policyformat(self.POLICY_HEAD)

        if value_head == "classical":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_CLASSICAL
            self.wdl = False
        elif value_head == "wdl":
            self.VALUE_HEAD = pb.NetworkFormat.VALUE_WDL
            self.wdl = True
        else:
            raise ValueError(
                "Unknown value head format: {}".format(value_head))

        self.net.set_valueformat(self.VALUE_HEAD)

        if moves_left_head == "none":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_NONE
            self.moves_left = False
        elif moves_left_head == "v1":
            self.MOVES_LEFT_HEAD = pb.NetworkFormat.MOVES_LEFT_V1
            self.moves_left = True
        else:
            raise ValueError(
                "Unknown moves left head format: {}".format(moves_left_head))

        self.net.set_movesleftformat(self.MOVES_LEFT_HEAD)

        if input_mode == "classic":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_CLASSICAL_112_PLANE
        elif input_mode == "frc_castling":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CASTLING_PLANE
        elif input_mode == "canonical":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION
        elif input_mode == "canonical_100":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES
        elif input_mode == "canonical_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON
        elif input_mode == "canonical_v2":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2
        elif input_mode == "canonical_v2_armageddon":
            self.INPUT_MODE = pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON
        else:
            raise ValueError(
                "Unknown input mode format: {}".format(input_mode))

        self.net.set_input(self.INPUT_MODE)

        self.swa_enabled = self.cfg['training'].get('swa', False)

        # Limit momentum of SWA exponential average to 1 - 1/(swa_max_n + 1)
        self.swa_max_n = self.cfg['training'].get('swa_max_n', 0)

        self.renorm_enabled = self.cfg['training'].get('renorm', False)
        self.renorm_max_r = self.cfg['training'].get('renorm_max_r', 1)
        self.renorm_max_d = self.cfg['training'].get('renorm_max_d', 0)
        self.renorm_momentum = self.cfg['training'].get(
            'renorm_momentum', 0.99)

        if self.cfg['gpu'] == 'all':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self.strategy = tf.distribute.MirroredStrategy()
            tf.distribute.experimental_set_strategy(self.strategy)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(gpus)
            tf.config.experimental.set_visible_devices(gpus[self.cfg['gpu']],
                                                       'GPU')
            tf.config.experimental.set_memory_growth(gpus[self.cfg['gpu']],
                                                     True)
            self.strategy = None
        if self.model_dtype == tf.float16:
            tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

        self.global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False,
                                       dtype=tf.int64)

        # ignore this -- from earlier discontinued testing on replacing residual stack
        # from Arcturus.constants import POS_ENC, W_CASTLE_OOO, W_CASTLE_OO, B_CASTLE_OOO, B_CASTLE_OO
        # self.cc = [W_CASTLE_OOO, W_CASTLE_OO, B_CASTLE_OOO, B_CASTLE_OO, POS_ENC]

    def init_v2(self, train_dataset, test_dataset, validation_dataset=None):
        if self.strategy is not None:
            self.train_dataset = self.strategy.experimental_distribute_dataset(
                train_dataset)
        else:
            self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)
        if self.strategy is not None:
            self.test_dataset = self.strategy.experimental_distribute_dataset(
                test_dataset)
        else:
            self.test_dataset = train_dataset
        self.test_iter = iter(self.test_dataset)
        if self.strategy is not None and validation_dataset is not None:
            self.validation_dataset = self.strategy.experimental_distribute_dataset(
                validation_dataset)
        else:
            self.validation_dataset = validation_dataset
        if self.strategy is not None:
            this = self
            with self.strategy.scope():
                this.init_net_v2()
        else:
            self.init_net_v2()

    def init_net_v2(self):
        self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
        input_var = tf.keras.Input(shape=(112, 8 * 8))
        x_planes = tf.keras.layers.Reshape([112, 8, 8])(input_var)
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            policy, value, moves_left, attn_wts = self.construct_net_v2(x_planes)
        else:
            policy, value, moves_left = self.construct_net_v2(x_planes)
        if self.moves_left:
            if self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
                outputs = [policy, value, moves_left, attn_wts]
            else:
                outputs = [policy, value, moves_left]
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            outputs = [policy, value, attn_wts]
        else:
            outputs = [policy, value]
        self.model = tf.keras.Model(inputs=input_var, outputs=outputs)
        # ignore this -- from earlier discontinued testing on replacing residual stack
        # import Arcturus.constants as c
        # import Arcturus.arcturus_model as am
        # self.model = am.Net(num_enc_layers=4, emb_size=512, d_model=1024, num_heads=8, dff=2048,
        #                     sq_emb=256, tcv=512, vh_layers=3, vh_width=128, tempreg=self.l2reg)
        # self.model = am.Net(num_enc_layers=8, emb_size=256, d_model=512, num_heads=8, dff=1024,
        #                     sq_emb=256, tcv=512, vh_layers=3, vh_width=128, tempreg=self.l2reg)
        # self.model = am.Net(num_enc_layers=8, emb_size=128, d_model=252, num_heads=12, dff=512,
        #                     sq_emb=128, tcv=512, vh_layers=3, vh_width=128, tempreg=self.l2reg)
        # self.model(tf.concat([c.POS_ENC, c.INIT_TOKENS], axis=2))
        print(self.model.summary())

        # swa_count initialized regardless to make checkpoint code simpler.
        self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
        self.swa_weights = None
        if self.swa_enabled:
            # Count of networks accumulated into SWA
            self.swa_weights = [
                tf.Variable(w, trainable=False) for w in self.model.weights
            ]

        self.active_lr = 0.01
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=lambda: self.active_lr, momentum=0.9, nesterov=True)
        self.orig_optimizer = self.optimizer
        if self.loss_scale != 1:
            self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                self.optimizer, self.loss_scale)

        def correct_policy(target, output):
            output = tf.cast(output, tf.float32)
            # Calculate loss on policy head
            if self.cfg['training'].get('mask_legal_moves'):
                # extract mask for legal moves from target policy
                move_is_legal = tf.greater_equal(target, 0)
                # replace logits of illegal moves with large negative value (so that it doesn't affect policy of legal moves) without gradient
                illegal_filler = tf.zeros_like(output) - 1.0e10
                output = tf.where(move_is_legal, output, illegal_filler)
            # y_ still has -1 on illegal moves, flush them to 0
            target = tf.nn.relu(target)
            return target, output

        def policy_loss(target, output):
            target, output = correct_policy(target, output)
            policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(target), logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_loss_fn = policy_loss

        def policy_accuracy(target, output):
            target, output = correct_policy(target, output)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.policy_accuracy_fn = policy_accuracy

        self.policy_accuracy_fn = policy_accuracy

        def moves_left_mean_error_fn(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(tf.abs(target - output))

        self.moves_left_mean_error = moves_left_mean_error_fn

        def policy_entropy(target, output):
            target, output = correct_policy(target, output)
            softmaxed = tf.nn.softmax(output)
            return tf.math.negative(
                tf.reduce_mean(
                    tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                                  axis=1)))

        self.policy_entropy_fn = policy_entropy

        def policy_uniform_loss(target, output):
            uniform = tf.where(tf.greater_equal(target, 0),
                               tf.ones_like(target), tf.zeros_like(target))
            balanced_uniform = uniform / tf.reduce_sum(
                uniform, axis=1, keepdims=True)
            target, output = correct_policy(target, output)
            policy_cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                                                        logits=output)
            return tf.reduce_mean(input_tensor=policy_cross_entropy)

        self.policy_uniform_loss_fn = policy_uniform_loss

        q_ratio = self.cfg['training'].get('q_ratio', 0)
        assert 0 <= q_ratio <= 1

        # Linear conversion to scalar to compute MSE with, for comparison to old values
        wdl = tf.expand_dims(tf.constant([1.0, 0.0, -1.0]), 1)

        self.qMix = lambda z, q: q * q_ratio + z * (1 - q_ratio)
        # Loss on value head
        if self.wdl:

            def value_loss(target, output):
                output = tf.cast(output, tf.float32)
                value_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(target), logits=output)
                return tf.reduce_mean(input_tensor=value_cross_entropy)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_z_conv = tf.matmul(tf.nn.softmax(output), wdl)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, scalar_z_conv))

            self.mse_loss_fn = mse_loss
        else:

            def value_loss(target, output):
                return tf.constant(0)

            self.value_loss_fn = value_loss

            def mse_loss(target, output):
                output = tf.cast(output, tf.float32)
                scalar_target = tf.matmul(target, wdl)
                return tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    scalar_target, output))

            self.mse_loss_fn = mse_loss

        if self.moves_left:

            def moves_left_loss(target, output):
                # Scale the loss to similar range as other losses.
                scale = 20.0
                target = target / scale
                output = tf.cast(output, tf.float32) / scale
                if self.strategy is not None:
                    huber = tf.keras.losses.Huber(
                        10.0 / scale, reduction=tf.keras.losses.Reduction.NONE)
                else:
                    huber = tf.keras.losses.Huber(10.0 / scale)
                return tf.reduce_mean(huber(target, output))
        else:
            moves_left_loss = None

        self.moves_left_loss_fn = moves_left_loss

        pol_loss_w = self.cfg['training']['policy_loss_weight']
        val_loss_w = self.cfg['training']['value_loss_weight']

        if self.moves_left:
            moves_loss_w = self.cfg['training']['moves_left_loss_weight']
        else:
            moves_loss_w = tf.constant(0.0, dtype=tf.float32)

        def _lossMix(policy, value, moves_left):
            return pol_loss_w * policy + val_loss_w * value + moves_loss_w * moves_left

        self.lossMix = _lossMix

        def accuracy(target, output):
            output = tf.cast(output, tf.float32)
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(input=target, axis=1),
                             tf.argmax(input=output, axis=1)), tf.float32))

        self.accuracy_fn = accuracy

        self.avg_policy_loss = []
        self.avg_value_loss = []
        self.avg_moves_left_loss = []
        self.avg_mse_loss = []
        self.avg_reg_term = []
        self.time_start = None
        self.last_steps = None
        # Set adaptive learning rate during training
        self.cfg['training']['lr_boundaries'].sort()
        self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
        self.lr = self.cfg['training']['lr_values'][0]
        self.test_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-test".format(self.cfg['name'])))
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(os.getcwd(),
                         "leelalogs/{}-train".format(self.cfg['name'])))
        if vars(self).get('validation_dataset', None) is not None:
            self.validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-validation".format(self.cfg['name'])))
        if self.swa_enabled:
            self.swa_writer = tf.summary.create_file_writer(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-swa-test".format(self.cfg['name'])))
            self.swa_validation_writer = tf.summary.create_file_writer(
                os.path.join(
                    os.getcwd(),
                    "leelalogs/{}-swa-validation".format(self.cfg['name'])))
        self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
                                              model=self.model,
                                              global_step=self.global_step,
                                              swa_count=self.swa_count)
        self.checkpoint.listed = self.swa_weights
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.root_dir,
            max_to_keep=50,
            keep_checkpoint_every_n_hours=24,
            checkpoint_name=self.cfg['name'])

    def replace_weights_v2(self, proto_filename, ignore_errors=False):
        self.net.parse_proto(proto_filename)

        filters, blocks = self.net.filters(), self.net.blocks()
        if not ignore_errors:
            if self.RESIDUAL_FILTERS != filters:
                raise ValueError("Number of filters doesn't match the network")
            if self.RESIDUAL_BLOCKS != blocks:
                raise ValueError("Number of blocks doesn't match the network")
            if self.POLICY_HEAD != self.net.pb.format.network_format.policy:
                raise ValueError("Policy head type doesn't match the network")
            if self.VALUE_HEAD != self.net.pb.format.network_format.value:
                raise ValueError("Value head type doesn't match the network")

        # List all tensor names we need weights for.
        names = []
        for weight in self.model.weights:
            names.append(weight.name)

        new_weights = self.net.get_weights_v2(names)
        for weight in self.model.weights:
            if 'renorm' in weight.name:
                # Renorm variables are not populated.
                continue

            try:
                new_weight = new_weights[weight.name]
            except KeyError:
                error_string = 'No values for tensor {} in protobuf'.format(
                    weight.name)
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if reduce(operator.mul, weight.shape.as_list(),
                      1) != len(new_weight):
                error_string = 'Tensor {} has wrong length. Tensorflow shape {}, size in protobuf {}'.format(
                    weight.name, weight.shape.as_list(), len(new_weight))
                if ignore_errors:
                    print(error_string)
                    continue
                else:
                    raise KeyError(error_string)

            if weight.shape.ndims == 4:
                # Rescale rule50 related weights as clients do not normalize the input.
                if weight.name == 'input/conv2d/kernel:0' and self.net.pb.format.network_format.input < pb.NetworkFormat.INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
                    num_inputs = 112
                    # 50 move rule is the 110th input, or 109 starting from 0.
                    rule50_input = 109
                    for i in range(len(new_weight)):
                        if (i % (num_inputs * 9)) // 9 == rule50_input:
                            new_weight[i] = new_weight[i] * 99

                # Convolution weights need a transpose
                #
                # TF (kYXInputOutput)
                # [filter_height, filter_width, in_channels, out_channels]
                #
                # Leela/cuDNN/Caffe (kOutputInputYX)
                # [output, input, filter_size, filter_size]
                s = weight.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[2, 3, 1, 0]))
            elif weight.shape.ndims == 2:
                # Fully connected layers are [in, out] in TF
                #
                # [out, in] in Leela
                #
                s = weight.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weight, shape=shape)
                weight.assign(tf.transpose(a=new_weight, perm=[1, 0]))
            else:
                # Biases, batchnorm etc
                new_weight = tf.constant(new_weight, shape=weight.shape)
                weight.assign(new_weight)
        # Replace the SWA weights as well, ensuring swa accumulation is reset.
        if self.swa_enabled:
            self.swa_count.assign(tf.constant(0.))
            self.update_swa_v2()
        # This should result in identical file to the starting one
        # self.save_leelaz_weights_v2('restored.pb.gz')

    def restore_v2(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def process_loop_v2(self, batch_size, test_batches, batch_splits=1):
        if self.swa_enabled:
            # split half of test_batches between testing regular weights and SWA weights
            test_batches //= 2
        # Make sure that ghost batch norm can be applied
        if self.virtual_batch_size and batch_size % self.virtual_batch_size != 0:
            # Adjust required batch size for batch splitting.
            required_factor = self.virtual_batch_size * self.cfg[
                'training'].get('num_batch_splits', 1)
            raise ValueError(
                'batch_size must be a multiple of {}'.format(required_factor))

        # Get the initial steps value in case this is a resume from a step count
        # which is not a multiple of total_steps.
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']
        for i in range(steps % total_steps, total_steps):
            print("step {}".format(i))
            self.process_v2(batch_size,
                            test_batches,
                            batch_splits=batch_splits)

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y, z, q, m):
        # ignore this -- from earlier discontinued testing on replacing residual stack
        ### CONVERT X TO TOKEN INPUT ###
        # x = self.planes_to_tokens(x)
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            policy = outputs[0]
            value = outputs[1]
            policy_loss = self.policy_loss_fn(y, policy)
            reg_term = sum(self.model.losses)
            if self.wdl:
                value_ce_loss = self.value_loss_fn(self.qMix(z, q), value)
                value_loss = value_ce_loss
            else:
                value_mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
                value_loss = value_mse_loss
            if self.moves_left:
                moves_left = outputs[2]
                moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            else:
                moves_left_loss = tf.constant(0.)
            total_loss = self.lossMix(policy_loss, value_loss,
                                      moves_left_loss) + reg_term
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
        if self.wdl:
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
        return policy_loss, value_loss, mse_loss, moves_left_loss, reg_term, tape.gradient(
            total_loss, self.model.trainable_weights)

    @tf.function()
    def strategy_process_inner_loop(self, x, y, z, q, m):
        # ignore this -- from earlier discontinued testing on replacing residual stack
        ### CONVERT X TO TOKEN INPUT ###
        # x = self.planes_to_tokens(x)
        policy_loss, value_loss, mse_loss, moves_left_loss, reg_term, new_grads = self.strategy.run(
            self.process_inner_loop, args=(x, y, z, q, m))
        policy_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                           policy_loss,
                                           axis=None)
        value_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          value_loss,
                                          axis=None)
        mse_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        mse_loss,
                                        axis=None)
        moves_left_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                               moves_left_loss,
                                               axis=None)
        reg_term = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        reg_term,
                                        axis=None)
        return policy_loss, value_loss, mse_loss, moves_left_loss, reg_term, new_grads

    def apply_grads(self, grads, effective_batch_splits):
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
            'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_weights))
        return grad_norm

    @tf.function()
    def strategy_apply_grads(self, grads, effective_batch_splits):
        grad_norm = self.strategy.run(self.apply_grads,
                                      args=(grads, effective_batch_splits))
        grad_norm = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         grad_norm,
                                         axis=None)
        return grad_norm

    @tf.function()
    def merge_grads(self, grads, new_grads):
        return [tf.math.add(a, b) for (a, b) in zip(grads, new_grads)]

    @tf.function()
    def strategy_merge_grads(self, grads, new_grads):
        return self.strategy.run(self.merge_grads, args=(grads, new_grads))

    def train_step(self, steps, batch_size, batch_splits):
        # need to add 1 to steps because steps will be incremented after gradient update
        if (steps +
            1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        # Run training for this batch
        grads = None
        for _ in range(batch_splits):
            x, y, z, q, m = next(self.train_iter)
            if self.strategy is not None:
                policy_loss, value_loss, mse_loss, moves_left_loss, reg_term, new_grads = self.strategy_process_inner_loop(
                    x, y, z, q, m)
            else:
                policy_loss, value_loss, mse_loss, moves_left_loss, reg_term, new_grads = self.process_inner_loop(
                    x, y, z, q, m)
            if not grads:
                grads = new_grads
            else:
                if self.strategy is not None:
                    grads = self.strategy_merge_grads(grads, new_grads)
                else:
                    grads = self.merge_grads(grads, new_grads)
            # Keep running averages
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss /= 4.0
            self.avg_policy_loss.append(policy_loss)
            if self.wdl:
                self.avg_value_loss.append(value_loss)
            if self.moves_left:
                self.avg_moves_left_loss.append(moves_left_loss)
            self.avg_mse_loss.append(mse_loss)
            self.avg_reg_term.append(reg_term)
        # Gradients of batch splits are summed, not averaged like usual, so need to scale lr accordingly to correct for this.
        effective_batch_splits = batch_splits
        if self.strategy is not None:
            effective_batch_splits = batch_splits * self.strategy.num_replicas_in_sync
        self.active_lr = self.lr / effective_batch_splits
        if self.strategy is not None:
            grad_norm = self.strategy_apply_grads(grads,
                                                  effective_batch_splits)
        else:
            grad_norm = self.apply_grads(grads, effective_batch_splits)

        # Note: grads variable at this point has not been unscaled or
        # had clipping applied. Since no code after this point depends
        # upon that it seems fine for now.

        # Update steps.
        self.global_step.assign_add(1)
        steps = self.global_step.read_value()

        if steps % self.cfg['training'][
            'train_avg_report_steps'] == 0 or steps % self.cfg['training'][
            'total_steps'] == 0:
            pol_loss_w = self.cfg['training']['policy_loss_weight']
            val_loss_w = self.cfg['training']['value_loss_weight']
            moves_loss_w = self.cfg['training']['moves_left_loss_weight']
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                steps_elapsed = steps - self.last_steps
                speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                                      elapsed)
            avg_policy_loss = np.mean(self.avg_policy_loss or [0])
            avg_moves_left_loss = np.mean(self.avg_moves_left_loss or [0])
            avg_value_loss = np.mean(self.avg_value_loss or [0])
            avg_mse_loss = np.mean(self.avg_mse_loss or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print(
                "step {}, lr={:g} policy={:g} value={:g} mse={:g} moves={:g} reg={:g} total={:g} ({:g} pos/s)"
                    .format(
                    steps, self.lr, avg_policy_loss, avg_value_loss,
                    avg_mse_loss, avg_moves_left_loss, avg_reg_term,
                    pol_loss_w * avg_policy_loss +
                    val_loss_w * avg_value_loss + avg_reg_term +
                    moves_loss_w * avg_moves_left_loss, speed))

            after_weights = self.read_weights()
            with self.train_writer.as_default():
                tf.summary.scalar("Policy Loss", avg_policy_loss, step=steps)
                tf.summary.scalar("Value Loss", avg_value_loss, step=steps)
                if self.moves_left:
                    tf.summary.scalar("Moves Left Loss",
                                      avg_moves_left_loss,
                                      step=steps)
                tf.summary.scalar("Reg term", avg_reg_term, step=steps)
                tf.summary.scalar("LR", self.lr, step=steps)
                tf.summary.scalar("Gradient norm",
                                  grad_norm / effective_batch_splits,
                                  step=steps)
                tf.summary.scalar("MSE Loss", avg_mse_loss, step=steps)
                self.compute_update_ratio_v2(before_weights, after_weights,
                                             steps)
            self.train_writer.flush()
            self.time_start = time_end
            self.last_steps = steps
            self.avg_policy_loss = []
            self.avg_moves_left_loss = []
            self.avg_value_loss = []
            self.avg_mse_loss = []
            self.avg_reg_term = []
        return steps

    def process_v2(self, batch_size, test_batches, batch_splits):
        # Get the initial steps value before we do a training step.
        steps = self.global_step.read_value()

        # By default disabled since 0 != 10.
        if steps % self.cfg['training'].get('profile_step_freq',
                                            1) == self.cfg['training'].get(
            'profile_step_offset', 10):
            self.profiling_start_step = steps
            tf.profiler.experimental.start(
                os.path.join(os.getcwd(),
                             "leelalogs/{}-profile".format(self.cfg['name'])))

        # Run test before first step to see delta since end of last run.
        if steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                # Steps is given as one higher than current in order to avoid it
                # being equal to the value the end of a run is stored against.
                self.calculate_test_summaries_v2(test_batches, steps + 1)
                if self.swa_enabled:
                    self.calculate_swa_summaries_v2(test_batches, steps + 1)

        # Determine learning rate
        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']
        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                                        tf.float32) / self.warmup_steps

        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps, batch_size, batch_splits)

        if self.swa_enabled and steps % self.cfg['training']['swa_steps'] == 0:
            self.update_swa_v2()

        # Calculate test values every 'test_steps', but also ensure there is
        # one at the final step so the delta to the first step can be calculted.
        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg[
            'training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries_v2(test_batches, steps)
                if self.swa_enabled:
                    self.calculate_swa_summaries_v2(test_batches, steps)

        if self.validation_dataset is not None and (
                steps % self.cfg['training']['validation_steps'] == 0
                or steps % self.cfg['training']['total_steps'] == 0):
            with tf.profiler.experimental.Trace("Validate", step_num=steps):
                if self.swa_enabled:
                    self.calculate_swa_validations_v2(steps)
                else:
                    self.calculate_test_validations_v2(steps)

        # Save session and weights at end, and also optionally every 'checkpoint_steps'.
        if steps % self.cfg['training']['total_steps'] == 0 or (
                'checkpoint_steps' in self.cfg['training']
                and steps % self.cfg['training']['checkpoint_steps'] == 0):
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(
                self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)
            swa_path = path + "-swa-" + str(evaled_steps)
            self.net.pb.training_params.training_steps = evaled_steps
            self.save_leelaz_weights_v2(leela_path)
            ###
            # self.model.save(leela_path)
            ###
            if self.swa_enabled:
                self.save_swa_weights_v2(swa_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                self.cfg['training'].get('profile_step_count', 0)
                or steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None

    def calculate_swa_summaries_v2(self, test_batches, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_test_writer, self.test_writer = self.test_writer, self.swa_writer
        print('swa', end=' ')
        self.calculate_test_summaries_v2(test_batches, steps)
        self.test_writer = true_test_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        outputs = self.model(x, training=False)
        policy = outputs[0]
        value = outputs[1]
        policy_loss = self.policy_loss_fn(y, policy)
        policy_accuracy = self.policy_accuracy_fn(y, policy)
        policy_entropy = self.policy_entropy_fn(y, policy)
        policy_ul = self.policy_uniform_loss_fn(y, policy)
        if self.wdl:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = self.accuracy_fn(self.qMix(z, q), value)
        else:
            value_loss = self.value_loss_fn(self.qMix(z, q), value)
            mse_loss = self.mse_loss_fn(self.qMix(z, q), value)
            value_accuracy = tf.constant(0.)
        if self.moves_left:
            moves_left = outputs[2]
            moves_left_loss = self.moves_left_loss_fn(m, moves_left)
            moves_left_mean_error = self.moves_left_mean_error(m, moves_left)
        else:
            moves_left_loss = tf.constant(0.)
            moves_left_mean_error = tf.constant(0.)
        return policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul

    @tf.function()
    def strategy_calculate_test_summaries_inner_loop(self, x, y, z, q, m):
        policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.strategy.run(
            self.calculate_test_summaries_inner_loop, args=(x, y, z, q, m))
        policy_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                           policy_loss,
                                           axis=None)
        value_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          value_loss,
                                          axis=None)
        mse_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        mse_loss,
                                        axis=None)
        policy_accuracy = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                               policy_accuracy,
                                               axis=None)
        value_accuracy = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                              value_accuracy,
                                              axis=None)
        moves_left_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                               moves_left_loss,
                                               axis=None)
        moves_left_mean_error = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, moves_left_mean_error, axis=None)
        policy_entropy = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                              policy_entropy,
                                              axis=None)
        policy_ul = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                         policy_ul,
                                         axis=None)
        return policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul

    # ignore this -- from earlier discontinued testing on replacing residual stack
    # @tf.function()
    # def planes_to_tokens(self, stack):
    #     our_pawns = stack[:, 0, :]
    #     our_knights = stack[:, 1, :]
    #     our_diagonals = stack[:, 2, :] + stack[:, 4, :]
    #     our_orthogonals = stack[:, 3, :] + stack[:, 4, :]
    #     our_kings = stack[:, 5, :]
    #
    #     their_pawns = stack[:, 6, :]
    #     their_knights = stack[:, 7, :]
    #     their_diagonals = stack[:, 8, :] + stack[:, 10, :]
    #     their_orthogonals = stack[:, 9, :] + stack[:, 10, :]
    #     their_kings = stack[:, 11, :]
    #
    #     our_pieces = tf.clip_by_value(our_pawns + our_knights + our_diagonals + our_orthogonals + our_kings,
    #                                   clip_value_min=0., clip_value_max=1.)
    #     their_pieces = tf.clip_by_value(their_pawns + their_knights + their_diagonals + their_orthogonals + their_kings,
    #                                     clip_value_min=0., clip_value_max=1.)
    #
    #     pawns = our_pawns + their_pawns
    #     knights = our_knights + their_knights
    #     diagonals = our_diagonals + their_diagonals
    #     orthogonals = our_orthogonals + their_orthogonals
    #     kings = our_kings + their_kings
    #
    #     tokens = tf.stack([our_pieces, their_pieces, pawns, knights, orthogonals, diagonals, kings], axis=2) * 3.
    #     batch_len = tf.shape(tokens)[0]
    #
    #     pos_enc = self.cc[4]
    #     pos_enc = tf.broadcast_to(pos_enc, [batch_len, 64, tf.shape(pos_enc)[2]])
    #     tokenized = tf.concat([pos_enc, tokens], axis=2)
    #
    #     def ones(x): return tf.greater(x, 0)
    #
    #     castling_us_ooo = ones(stack[:, 104, 0])
    #     castling_us_ooo = tf.logical_and(tf.reshape(castling_us_ooo, [-1, 1]),
    #                                      tf.broadcast_to(tf.greater(self.cc[0], 0), [batch_len, 64]))
    #     castling_us_oo = ones(stack[:, 105, 0])
    #     castling_us_oo = tf.logical_and(tf.reshape(castling_us_oo, [-1, 1]),
    #                                     tf.broadcast_to(tf.greater(self.cc[1], 0), [batch_len, 64]))
    #     castling_them_ooo = ones(stack[:, 104, 56])
    #     castling_them_ooo = tf.logical_and(tf.reshape(castling_them_ooo, [-1, 1]),
    #                                        tf.broadcast_to(tf.greater(self.cc[2], 0), [batch_len, 64]))
    #     castling_them_oo = ones(stack[:, 105, 56])
    #     castling_them_oo = tf.logical_and(tf.reshape(castling_them_oo, [-1, 1]),
    #                                       tf.broadcast_to(tf.greater(self.cc[3], 0), [batch_len, 64]))
    #
    #     castle_us = tf.logical_or(castling_us_ooo, castling_us_oo)
    #     castle_them = tf.logical_or(castling_them_ooo, castling_them_oo)
    #     castling = tf.logical_or(castle_us, castle_them)
    #
    #     their_pawns_r7 = ones(their_pawns[:, 48:56])
    #     their_pawns_r5 = ones(their_pawns[:, 32:40])
    #     their_pawns_prev_r7 = ones(stack[:, 19, 48:56])
    #     their_pawns_prev_r5 = ones(stack[:, 19, 32:40])
    #
    #     ep_file = tf.logical_and(tf.math.logical_xor(their_pawns_r5, their_pawns_prev_r5),
    #                              tf.math.logical_xor(their_pawns_r7, their_pawns_prev_r7))
    #     ep = tf.concat([tf.greater(tf.zeros([batch_len, 40]), 0),
    #                     ep_file,
    #                     tf.greater(tf.zeros([batch_len, 16]), 0)],
    #                    axis=1)
    #
    #     spec_move = tf.zeros_like(pawns)
    #     spec_move_filter = tf.logical_or(castling, ep)
    #     legal_spec = tf.ones_like(pawns)
    #
    #     spec_move = tf.where(spec_move_filter, legal_spec, spec_move)
    #
    #     return tf.concat([tokenized, tf.expand_dims(spec_move, axis=2)], axis=2)
    ##########

    def calculate_test_summaries_v2(self, test_batches, steps):
        sum_policy_accuracy = 0
        sum_value_accuracy = 0
        sum_moves_left = 0
        sum_moves_left_mean_error = 0
        sum_mse = 0
        sum_policy = 0
        sum_value = 0
        sum_policy_entropy = 0
        sum_policy_ul = 0
        for _ in range(0, test_batches):
            x, y, z, q, m = next(self.test_iter)
            # ignore this -- from earlier discontinued testing on replacing residual stack
            ### CONVERT X TO TOKEN INPUT ###
            # x = self.planes_to_tokens(x)
            if self.strategy is not None:
                policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            sum_policy_accuracy += policy_accuracy
            sum_policy_entropy += policy_entropy
            sum_policy_ul += policy_ul
            sum_mse += mse_loss
            sum_policy += policy_loss
            if self.wdl:
                sum_value_accuracy += value_accuracy
                sum_value += value_loss
            if self.moves_left:
                sum_moves_left += moves_left_loss
                sum_moves_left_mean_error += moves_left_mean_error
        sum_policy_accuracy /= test_batches
        sum_policy_accuracy *= 100
        sum_policy /= test_batches
        sum_policy_entropy /= test_batches
        sum_policy_ul /= test_batches
        sum_value /= test_batches
        if self.wdl:
            sum_value_accuracy /= test_batches
            sum_value_accuracy *= 100
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * test_batches)
        if self.moves_left:
            sum_moves_left /= test_batches
            sum_moves_left_mean_error /= test_batches
        self.net.pb.training_params.learning_rate = self.lr
        self.net.pb.training_params.mse_loss = sum_mse
        self.net.pb.training_params.policy_loss = sum_policy
        # TODO store value and value accuracy in pb
        self.net.pb.training_params.accuracy = sum_policy_accuracy
        with self.test_writer.as_default():
            tf.summary.scalar("Policy Loss", sum_policy, step=steps)
            tf.summary.scalar("Value Loss", sum_value, step=steps)
            tf.summary.scalar("MSE Loss", sum_mse, step=steps)
            tf.summary.scalar("Policy Accuracy",
                              sum_policy_accuracy,
                              step=steps)
            tf.summary.scalar("Policy Entropy", sum_policy_entropy, step=steps)
            tf.summary.scalar("Policy UL", sum_policy_ul, step=steps)
            if self.wdl:
                tf.summary.scalar("Value Accuracy",
                                  sum_value_accuracy,
                                  step=steps)
            if self.moves_left:
                tf.summary.scalar("Moves Left Loss",
                                  sum_moves_left,
                                  step=steps)
                tf.summary.scalar("Moves Left Mean Error",
                                  sum_moves_left_mean_error,
                                  step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()

        print(
            "step {}, policy={:g} value={:g} policy accuracy={:g}% value accuracy={:g}% mse={:g} policy entropy={:g} policy ul={:g}". \
            format(steps, sum_policy, sum_value, sum_policy_accuracy, sum_value_accuracy, sum_mse, sum_policy_entropy,
                   sum_policy_ul), end='')

        if self.moves_left:
            print(" moves={:g} moves mean={:g}".format(
                sum_moves_left, sum_moves_left_mean_error))
        else:
            print()

    def calculate_swa_validations_v2(self, steps):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        true_validation_writer, self.validation_writer = self.validation_writer, self.swa_validation_writer
        print('swa', end=' ')
        self.calculate_test_validations_v2(steps)
        self.validation_writer = true_validation_writer
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def calculate_test_validations_v2(self, steps):
        sum_policy_accuracy = 0
        sum_value_accuracy = 0
        sum_moves_left = 0
        sum_moves_left_mean_error = 0
        sum_mse = 0
        sum_policy = 0
        sum_value = 0
        sum_policy_entropy = 0
        sum_policy_ul = 0
        counter = 0
        for (x, y, z, q, m) in self.validation_dataset:
            # ignore this -- from earlier discontinued testing on replacing residual stack
            ### CONVERT X TO TOKEN INPUT ###
            # x = self.planes_to_tokens(x)
            if self.strategy is not None:
                policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.strategy_calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            else:
                policy_loss, value_loss, moves_left_loss, mse_loss, policy_accuracy, value_accuracy, moves_left_mean_error, policy_entropy, policy_ul = self.calculate_test_summaries_inner_loop(
                    x, y, z, q, m)
            sum_policy_accuracy += policy_accuracy
            sum_policy_entropy += policy_entropy
            sum_policy_ul += policy_ul
            sum_mse += mse_loss
            sum_policy += policy_loss
            if self.moves_left:
                sum_moves_left += moves_left_loss
                sum_moves_left_mean_error += moves_left_mean_error
            counter += 1
            if self.wdl:
                sum_value_accuracy += value_accuracy
                sum_value += value_loss
        sum_policy_accuracy /= counter
        sum_policy_accuracy *= 100
        sum_policy /= counter
        sum_policy_entropy /= counter
        sum_policy_ul /= counter
        sum_value /= counter
        if self.wdl:
            sum_value_accuracy /= counter
            sum_value_accuracy *= 100
        if self.moves_left:
            sum_moves_left /= counter
            sum_moves_left_mean_error /= counter
        # Additionally rescale to [0, 1] so divide by 4
        sum_mse /= (4.0 * counter)
        with self.validation_writer.as_default():
            tf.summary.scalar("Policy Loss", sum_policy, step=steps)
            tf.summary.scalar("Value Loss", sum_value, step=steps)
            tf.summary.scalar("MSE Loss", sum_mse, step=steps)
            tf.summary.scalar("Policy Accuracy",
                              sum_policy_accuracy,
                              step=steps)
            tf.summary.scalar("Policy Entropy", sum_policy_entropy, step=steps)
            tf.summary.scalar("Policy UL", sum_policy_ul, step=steps)
            if self.wdl:
                tf.summary.scalar("Value Accuracy",
                                  sum_value_accuracy,
                                  step=steps)
            if self.moves_left:
                tf.summary.scalar("Moves Left Loss",
                                  sum_moves_left,
                                  step=steps)
                tf.summary.scalar("Moves Left Mean Error",
                                  sum_moves_left_mean_error,
                                  step=steps)
        self.validation_writer.flush()

        print(
            "step {}, validation: policy={:g} value={:g} policy accuracy={:g}% value accuracy={:g}% mse={:g} policy entropy={:g} policy ul={:g}". \
            format(steps, sum_policy, sum_value, sum_policy_accuracy, sum_value_accuracy, sum_mse, sum_policy_entropy,
                   sum_policy_ul), end='')

        if self.moves_left:
            print(" moves={:g} moves mean={:g}".format(
                sum_moves_left, sum_moves_left_mean_error))
        else:
            print()

    @tf.function()
    def compute_update_ratio_v2(self, before_weights, after_weights, steps):
        """Compute the ratio of gradient norm to weight norm.

        Adapted from https://github.com/tensorflow/minigo/blob/c923cd5b11f7d417c9541ad61414bf175a84dc31/dual_net.py#L567
        """
        deltas = [
            after - before
            for after, before in zip(after_weights, before_weights)
        ]
        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
            tf.math.reduce_euclidean_norm(w) for w in before_weights
        ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                  for d, w, tensor in zip(delta_norms, weight_norms,
                                          self.model.weights)
                  if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        # Filtering is hard, so just push infinities/NaNs to an unreasonably large value.
        ratios = [
            tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
        ]
        tf.summary.histogram('update_ratios_log10',
                             tf.stack(ratios),
                             buckets=1000,
                             step=steps)

    def update_swa_v2(self):
        num = self.swa_count.read_value()
        for (w, swa) in zip(self.model.weights, self.swa_weights):
            swa.assign(swa.read_value() * (num / (num + 1.)) + w.read_value() *
                       (1. / (num + 1.)))
        self.swa_count.assign(min(num + 1., self.swa_max_n))

    def save_swa_weights_v2(self, filename):
        backup = self.read_weights()
        for (swa, w) in zip(self.swa_weights, self.model.weights):
            w.assign(swa.read_value())
        self.save_leelaz_weights_v2(filename)
        ###
        # self.model.save(filename)
        ###
        for (old, w) in zip(backup, self.model.weights):
            w.assign(old)

    def save_leelaz_weights_v2(self, filename):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        self.net.save_proto(filename)

    def batch_norm_v2(self, input, name, scale=False):
        if self.renorm_enabled:
            clipping = {
                "rmin": 1.0 / self.renorm_max_r,
                "rmax": self.renorm_max_r,
                "dmax": self.renorm_max_d
            }
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                fused=False,
                center=True,
                scale=scale,
                renorm=True,
                renorm_clipping=clipping,
                renorm_momentum=self.renorm_momentum,
                name=name)(input)
        else:
            return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=1,
                center=True,
                scale=scale,
                virtual_batch_size=self.virtual_batch_size,
                name=name)(input)

    def squeeze_excitation_v2(self, inputs, channels, name):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(
            data_format='channels_first')(inputs)
        squeezed = tf.keras.layers.Activation('relu')(tf.keras.layers.Dense(
            channels // self.SE_ratio,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            name=name + '/se/dense1')(pooled))
        excited = tf.keras.layers.Dense(2 * channels,
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=self.l2reg,
                                        name=name + '/se/dense2')(squeezed)
        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block_v2(self,
                      inputs,
                      filter_size,
                      output_channels,
                      name,
                      bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels,
                                      filter_size,
                                      use_bias=False,
                                      padding='same',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      data_format='channels_first',
                                      name=name + '/conv2d')(inputs)
        return tf.keras.layers.Activation('relu')(self.batch_norm_v2(
            conv, name=name + '/bn', scale=bn_scale))

    def residual_block_v2(self, inputs, channels, name):
        conv1 = tf.keras.layers.Conv2D(channels,
                                       3,
                                       use_bias=False,
                                       padding='same',
                                       kernel_initializer='glorot_normal',
                                       kernel_regularizer=self.l2reg,
                                       data_format='channels_first',
                                       name=name + '/1/conv2d')(inputs)
        out1 = tf.keras.layers.Activation('relu')(self.batch_norm_v2(
            conv1, name + '/1/bn', scale=False))
        conv2 = tf.keras.layers.Conv2D(channels,
                                       3,
                                       use_bias=False,
                                       padding='same',
                                       kernel_initializer='glorot_normal',
                                       kernel_regularizer=self.l2reg,
                                       data_format='channels_first',
                                       name=name + '/2/conv2d')(out1)
        out2 = self.squeeze_excitation_v2(self.batch_norm_v2(conv2,
                                                             name + '/2/bn',
                                                             scale=True),
                                          channels,
                                          name=name + '/se')
        return tf.keras.layers.Activation('relu')(tf.keras.layers.add(
            [inputs, out2]))

    ### THE FOLLOWING IS NEW CODE FOR ATTENTION AND ENCODER LAYERS ###
    @staticmethod
    def scaled_dot_product_attention(q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], k.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    @staticmethod
    def split_heads(inputs, batch_size, num_heads, depth):
        if num_heads < 2:
            return inputs
        reshaped = tf.reshape(inputs, (batch_size, -1, num_heads, depth))
        return tf.transpose(reshaped, perm=[0, 2, 1, 3]) #(batch_size, num_heads, seq_len, depth)

    # multi-head attention in encoder layers
    def mha(self, inputs, emb_size, d_model, num_heads, name):
        assert d_model % num_heads == 0
        depth = d_model // num_heads
        q = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                  name=name + '/wq')(inputs)
        k = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                  name=name + '/wk')(inputs)
        v = tf.keras.layers.Dense(d_model, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                  name=name + '/wv')(inputs)
        batch_size = tf.shape(q)[0]
        q = self.split_heads(q, batch_size, num_heads, depth)
        k = self.split_heads(k, batch_size, num_heads, depth)
        v = self.split_heads(v, batch_size, num_heads, depth)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        if num_heads > 1:
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, d_model))  # concatenate heads
        output = tf.keras.layers.Dense(emb_size, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                       name=name + "/dense")(scaled_attention)
        return output, attention_weights

    def ffn(self, inputs, emb_size, dff, name):
        dense1 = tf.keras.layers.Dense(dff, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                       activation='selu', name=name + "/dense1")(inputs)
        return tf.keras.layers.Dense(emb_size, kernel_initializer='glorot_normal', kernel_regularizer=self.l2reg,
                                     name=name + "/dense2")(dense1)

    def encoder_layer(self, inputs, emb_size, d_model, num_heads, dff, name, rate=0.1, training=False):
        attn_output, attn_wts = self.mha(inputs, emb_size, d_model, num_heads, name=name + "/mha")
        # skip connection
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "/ln1")(inputs + attn_output)
        # dff is the hidden layer size, emb_size is the input and output channel size of the encoder layer
        ffn_output = self.ffn(out1, emb_size, dff, name=name + "/ffn")
        # skip connection
        out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=name + "/ln2")(out1 + ffn_output)
        return out2, attn_wts

    def construct_net_v2(self, inputs):
        flow = self.conv_block_v2(inputs,
                                  filter_size=3,
                                  output_channels=self.RESIDUAL_FILTERS,
                                  name='input',
                                  bn_scale=True)
        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block_v2(flow,
                                          self.RESIDUAL_FILTERS,
                                          name='residual_{}'.format(i + 1))

        # Policy head
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_CONVOLUTION:
            conv_pol = self.conv_block_v2(
                flow,
                filter_size=3,
                output_channels=self.RESIDUAL_FILTERS,
                name='policy1')
            conv_pol2 = tf.keras.layers.Conv2D(
                80,
                3,
                use_bias=True,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                bias_regularizer=self.l2reg,
                data_format='channels_first',
                name='policy')(conv_pol)
            h_fc1 = ApplyPolicyMap()(conv_pol2)  # 80x8x8
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_CLASSICAL:
            conv_pol = self.conv_block_v2(flow,
                                          filter_size=1,
                                          output_channels=self.policy_channels,
                                          name='policy')
            h_conv_pol_flat = tf.keras.layers.Flatten()(conv_pol)
            h_fc1 = tf.keras.layers.Dense(1858,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name='policy/dense')(h_conv_pol_flat)
        ### SELF-ATTENTION POLICY ###
        elif self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            # transpose and reshape
            tokens = tf.transpose(flow, perm=[0, 2, 3, 1])
            tokens = tf.reshape(tokens, [-1, 64, self.RESIDUAL_FILTERS])

            # 'embedding' layer: found to increase training perf., but using multiple does not increase perf. further
            tokens = tf.keras.layers.Dense(self.emb_size_pol, kernel_initializer='glorot_normal',
                                           kernel_regularizer=self.l2reg, activation='selu',
                                           name='policy/embedding')(tokens)

            """PAWN PROMOTION FIRST CONCEPT, DOES NOT WORK WITH ENCODER LAYERS YET"""
            # r8 = tokens[:, -8:, :]
            # r8q = tf.keras.layers.Dense(self.emb_size_pol, kernel_initializer='glorot_normal',
            #                             kernel_regularizer=self.l2reg, activation='selu',
            #                             name='policy/embedding/pp_queen')(r8)
            # r8r = tf.keras.layers.Dense(self.emb_size_pol, kernel_initializer='glorot_normal',
            #                             kernel_regularizer=self.l2reg, activation='selu',
            #                             name='policy/embedding/pp_rook')(r8)
            # r8b = tf.keras.layers.Dense(self.emb_size_pol, kernel_initializer='glorot_normal',
            #                             kernel_regularizer=self.l2reg, activation='selu',
            #                             name='policy/embedding/pp_bishop')(r8)
            # key_tokens = tf.concat([tokens, r8b, r8r, r8q], axis=1)

            """ENCODER LAYERS (default none: no discovered performance benefit yet, large hit to speed)"""
            # resid = tokens  # for global skip connection, untested but was helpful when testing on transformer body
            attn_wts = []
            for i in range(self.enc_layers_pol):
                tokens, attn_wts_l = self.encoder_layer(tokens,
                                                        self.emb_size_pol, self.d_model_pol_enc, self.n_heads_pol_enc,
                                                        self.dff_pol_enc, name='policy/enc_layer/' + str(i),
                                                        rate=0.1, training=True)
                attn_wts.append(attn_wts_l)
                # for global skip connection
                # tokens = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='policy/global_ln/' + str(i))\
                #     ((1/tf.math.log(i+1.718282))*resid + tokens)

            # TODO: test only computing query vectors for "our" pieces for performance improvement
            q = tf.keras.layers.Dense(self.d_model_pol_hd,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      name='policy/attention/wq')(tokens)
            # for pawn promotion concept, change input for this layer from tokens -> key_tokens
            k = tf.keras.layers.Dense(self.d_model_pol_hd,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      name='policy/attention/wk')(tokens)

            # split heads, does nothing if n_heads_pol_hd is 1
            assert self.d_model_pol_hd % self.n_heads_pol_hd == 0
            depth = self.d_model_pol_hd // self.n_heads_pol_hd
            batch_size = tf.shape(q)[0]
            q = self.split_heads(q, batch_size, self.n_heads_pol_hd, depth)
            k = self.split_heads(k, batch_size, self.n_heads_pol_hd, depth)

            # compute policy logits
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], k.dtype)
            policy_attn_logits = matmul_qk / tf.math.sqrt(dk)

            # summarize the policy logits from all heads, using one of several methods
            if self.n_heads_pol_hd > 1:
                attn_wts.append(policy_attn_logits)
                """
                ARITHMETIC MEAN ACROSS ALL HEADS
                tested, but only a small performance benefit found so far, and only for 2 heads
                w/2 heads, little to no hit to training speed, 4 & 8 heads saw a small hit to speed
                >2 heads may work for larger nets, trained for longer, and with larger d_model sizes
                """
                policy_attn_logits = tf.math.reduce_mean(policy_attn_logits, axis=1)
                """SUM ACROSS ALL HEADS (untested)"""
                # policy_attn_logits = tf.reduce_sum(policy_attn_logits, axis=1)
                """GEOMETRIC MEAN ACROSS ALL HEADS (crazy idea, untested)"""
                # signs = tf.reduce_prod(tf.math.sign(policy_attn_logits), axis=1)
                # policy_attn_logits = tf.math.abs(policy_attn_logits)
                # USE THIS #
                # policy_attn_logits = tf.math.exp(tf.math.reduce_mean(tf.math.log(policy_attn_logits), axis=1))
                # OR THIS #
                # policy_attn_logits = tf.math.pow(tf.math.reduce_prod(policy_attn_logits, axis=1),
                #                                  1/self.n_heads_pol_hd)  # this method is probably faster and more
                #                                                          # precise, but carries with it the small risk
                #                                                          # of Inf values when summarizing many heads
                # BEFORE THIS #
                # policy_attn_logits = tf.math.multiply(policy_attn_logits, signs)
            attn_wts.append(policy_attn_logits)

            # TODO (after previous TODO): re-embed queries in original squares so output is BATCH_SIZEx64x64

            # apply the new policy map so output becomes (BATCH_SIZE, 1856)
            h_fc1 = ApplyAttentionPolicyMap()(policy_attn_logits)

        else:
            raise ValueError("Unknown policy head type {}".format(
                self.POLICY_HEAD))

        # Value head
        conv_val = self.conv_block_v2(flow,
                                      filter_size=1,
                                      output_channels=32,
                                      name='value')
        h_conv_val_flat = tf.keras.layers.Flatten()(conv_val)
        h_fc2 = tf.keras.layers.Dense(128,
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=self.l2reg,
                                      activation='relu',
                                      name='value/dense1')(h_conv_val_flat)
        if self.wdl:
            h_fc3 = tf.keras.layers.Dense(3,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          bias_regularizer=self.l2reg,
                                          name='value/dense2')(h_fc2)
        else:
            h_fc3 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='tanh',
                                          name='value/dense2')(h_fc2)

        # Moves left head
        if self.moves_left:
            conv_mov = self.conv_block_v2(flow,
                                          filter_size=1,
                                          output_channels=8,
                                          name='moves_left')
            h_conv_mov_flat = tf.keras.layers.Flatten()(conv_mov)
            h_fc4 = tf.keras.layers.Dense(
                128,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                activation='relu',
                name='moves_left/dense1')(h_conv_mov_flat)

            h_fc5 = tf.keras.layers.Dense(1,
                                          kernel_initializer='glorot_normal',
                                          kernel_regularizer=self.l2reg,
                                          activation='relu',
                                          name='moves_left/dense2')(h_fc4)
        else:
            h_fc5 = None

        # attention weights added as additional output for visualization script -- not necessary for engine to perform
        if self.POLICY_HEAD == pb.NetworkFormat.POLICY_ATTENTION:
            return h_fc1, h_fc3, h_fc5, attn_wts
        return h_fc1, h_fc3, h_fc5


# some old testing code
def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))
    tfprocess = TFProcess(cfg)
    tfprocess.init_net_v2()
    # tfprocess.restore_v2()
    model = tfprocess.model
    model.summary()


if __name__ == "__main__":
    import yaml
    import argparse

    argparser = argparse.ArgumentParser(description= \
                                            'Tensorflow pipeline for training Leela Chess.')
    argparser.add_argument('--cfg',
                           type=argparse.FileType('r'),
                           help='yaml configuration with training parameters')
    argparser.add_argument('--output',
                           type=str,
                           help='file to store weights in')

    # mp.set_start_method('spawn')
    main(argparser.parse_args())

import os
import numpy as np
import pandas as pd
import json
from torch.utils.tensorboard import SummaryWriter
import time


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Metrics(object):

    def __init__(self, clients, options, name='', append2suffix=None, result_prefix='./result', train_metric_extend_columns=None, test_metric_extend_columns=None):
        self.options = options
        num_rounds = options['num_rounds'] + 1

        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}


        if train_metric_extend_columns is not None:
            assert isinstance(train_metric_extend_columns, list)
        else:
            train_metric_extend_columns = []
        self.train_metric_writer = pd.DataFrame(columns=['loss', 'acc'] + train_metric_extend_columns)


        if test_metric_extend_columns is not None:
            assert isinstance(test_metric_extend_columns, list)
        else:
            test_metric_extend_columns = []
        self.test_metric_writer = pd.DataFrame(columns=['loss', 'acc'] + test_metric_extend_columns)
        # 记录训练的信息
        # customs
        self.customs_data = dict()
        self.num_rounds = num_rounds
        self.result_path = mkdir(os.path.join(result_prefix, self.options['dataset']))

        suffix = '{}_sd{}_lr{}_ep{}_bs{}_wd{}'.format(name,
                                                      options['seed'],
                                                      options['lr'],
                                                      options['num_epochs'],
                                                      options['batch_size'],
                                                      options['wd'])
        if append2suffix is not None:
            suffix += '_' + append2suffix

        self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algo'],
                                             options['model'], suffix)
        # if options['dis']:
        #     suffix = options['dis']
        #     self.exp_name += '_{}'.format(suffix)
        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        eval_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        test_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'test.event'))
        self.eval_metric_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval_metric'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(eval_event_folder)
        self.test_writer = SummaryWriter(test_event_folder)

    # 更新通信统计信息，包括字节数、计算量和读取字节数。这通常与客户端的通信相关。
    def update_commu_stats(self, round_i, stats):
        cid, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']

        self.bytes_written[cid][round_i] += bytes_w
        self.client_computations[cid][round_i] += comp
        self.bytes_read[cid][round_i] += bytes_r

    # 扩展通信统计信息，允许一次更新多个客户端的统计信息。
    def extend_commu_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_commu_stats(round_i, stats)


    def update_grads_stats(self, round_i, stats):
        self.gradnorm_on_train_data[round_i] = stats['gradnorm']
        self.graddiff_on_train_data[round_i] = stats['graddiff']
        self.train_writer.add_scalar('gradnorm', stats['gradnorm'], round_i)
        self.train_writer.add_scalar('graddiff', stats['graddiff'], round_i)

    # 更新训练过程中的统计信息，主要包括训练损失（loss）和准确度（accuracy）。这些信息被记录到训练数据的DataFrame中。
    def update_train_stats_only_acc_loss(self, round_i, train_stats):
        self.train_metric_writer = self.train_metric_writer.append(train_stats, ignore_index=True)

        for k, v in train_stats.items():
            self.train_writer.add_scalar('train' + k, v, round_i)

    def update_test_stats_only_acc_loss(self, round_i, test_stats):
        self.test_metric_writer = self.test_metric_writer.append(test_stats, ignore_index=True)
        for k, v in test_stats.items():
            self.test_writer.add_scalar('test' + k, v, round_i)

   # 更新评估过程中的统计信息，将这些信息写入文件以便后续分析。
    def update_eval_stats(self, round_i, df, on_which, filename, other_to_logger):
        df.to_csv(os.path.join(self.eval_metric_folder, filename))
        for k, v in other_to_logger.items():
            self.eval_writer.add_scalar(f'on_{on_which}_{k}', v, round_i)

    # 更新自定义标量信息，这些信息可根据需要添加。例如，可以记录训练过程中的自定义指标。
    def update_custom_scalars(self, round_i, **data):
        for key, scalar in data.items():
            if key not in self.customs_data:
                self.customs_data[key] = [0] * self.num_rounds
            self.customs_data[key][round_i] = scalar
            self.train_writer.add_scalar(key, scalar_value=scalar, global_step=round_i)

    # 将所有记录的统计信息和指标写入文件，以便离线分析
    def write(self):
        metrics = dict()

        # Dict(key=cid, value=list(stats for each round))
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        for key, data in self.customs_data.items():
            metrics[key] = data
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')
        params_dir = os.path.join(self.result_path, self.exp_name, 'params.json')
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)

        with open(params_dir, 'w') as ouf:
            json.dump(self.options, ouf)

        self.train_metric_writer.to_csv(os.path.join(self.result_path, self.exp_name, 'train_metric.csv'))
        self.test_metric_writer.to_csv(os.path.join(self.result_path, self.exp_name, 'test_metric.csv'))

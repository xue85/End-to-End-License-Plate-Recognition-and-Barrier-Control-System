# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import paddle
import paddle.distributed as dist

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_re_results
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps, print_dict
from tools.program import ArgsParser, load_config, merge_config, check_gpu
from tools.infer_vqa_token_ser import SerPredictor


class ReArgsParser(ArgsParser):
    def __init__(self):
        super(ReArgsParser, self).__init__()
        self.add_argument(
            "-c_ser", "--config_ser", help="ser configuration file to use")
        self.add_argument(
            "-o_ser",
            "--opt_ser",
            nargs='+',
            help="set ser configuration options ")

    def parse_args(self, argv=None):
        args = super(ReArgsParser, self).parse_args(argv)
        assert args.config_ser is not None, \
            "Please specify --config_ser=ser_configure_file_path."
        args.opt_ser = self._parse_opt(args.opt_ser)
        return args


def make_input(ser_inputs, ser_results):
    entities_labels = {'HEADER': 0, 'QUESTION': 1, 'ANSWER': 2}

    entities = ser_inputs[8][0]
    ser_results = ser_results[0]
    assert len(entities) == len(ser_results)

    # entities
    start = []
    end = []
    label = []
    entity_idx_dict = {}
    for i, (res, entity) in enumerate(zip(ser_results, entities)):
        if res['pred'] == 'O':
            continue
        entity_idx_dict[len(start)] = i
        start.append(entity['start'])
        end.append(entity['end'])
        label.append(entities_labels[res['pred']])
    entities = dict(start=start, end=end, label=label)

    # relations
    head = []
    tail = []
    for i in range(len(entities["label"])):
        for j in range(len(entities["label"])):
            if entities["label"][i] == 1 and entities["label"][j] == 2:
                head.append(i)
                tail.append(j)

    relations = dict(head=head, tail=tail)

    batch_size = ser_inputs[0].shape[0]
    entities_batch = []
    relations_batch = []
    entity_idx_dict_batch = []
    for b in range(batch_size):
        entities_batch.append(entities)
        relations_batch.append(relations)
        entity_idx_dict_batch.append(entity_idx_dict)

    ser_inputs[8] = entities_batch
    ser_inputs.append(relations_batch)
    # remove ocr_info segment_offset_id and label in ser input
    ser_inputs.pop(7)
    ser_inputs.pop(6)
    ser_inputs.pop(1)
    return ser_inputs, entity_idx_dict_batch


class SerRePredictor(object):
    def __init__(self, config, ser_config):
        self.ser_engine = SerPredictor(ser_config)

        #  init re model 
        global_config = config['Global']

        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)

        # build model
        self.model = build_model(config['Architecture'])

        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        self.model.eval()

    def __call__(self, img_path):
        ser_results, ser_inputs = self.ser_engine(img_path)
        paddle.save(ser_inputs, 'ser_inputs.npy')
        paddle.save(ser_results, 'ser_results.npy')
        re_input, entity_idx_dict_batch = make_input(ser_inputs, ser_results)
        preds = self.model(re_input)
        post_result = self.post_process_class(
            preds,
            ser_results=ser_results,
            entity_idx_dict_batch=entity_idx_dict_batch)
        return post_result


def preprocess():
    FLAGS = ReArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)

    ser_config = load_config(FLAGS.config_ser)
    ser_config = merge_config(ser_config, FLAGS.opt_ser)

    logger = get_logger()

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    device = device = 'gpu:{}'.format(dist.get_rank()) if use_gpu else 'cpu'
    device = paddle.set_device(device)

    logger.info('{} re config {}'.format('*' * 10, '*' * 10))
    print_dict(config, logger)
    logger.info('\n')
    logger.info('{} ser config {}'.format('*' * 10, '*' * 10))
    print_dict(ser_config, logger)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            device))
    return config, ser_config, device, logger


if __name__ == '__main__':
    config, ser_config, device, logger = preprocess()
    os.makedirs(config['Global']['save_res_path'], exist_ok=True)

    ser_re_engine = SerRePredictor(config, ser_config)

    infer_imgs = get_image_file_list(config['Global']['infer_img'])
    with open(
            os.path.join(config['Global']['save_res_path'],
                         "infer_results.txt"),
            "w",
            encoding='utf-8') as fout:
        for idx, img_path in enumerate(infer_imgs):
            save_img_path = os.path.join(
                config['Global']['save_res_path'],
                os.path.splitext(os.path.basename(img_path))[0] + "_ser.jpg")
            logger.info("process: [{}/{}], save result to {}".format(
                idx, len(infer_imgs), save_img_path))

            result = ser_re_engine(img_path)
            result = result[0]
            fout.write(img_path + "\t" + json.dumps(
                {
                    "ser_result": result,
                }, ensure_ascii=False) + "\n")
            img_res = draw_re_results(img_path, result)
            cv2.imwrite(save_img_path, img_res)

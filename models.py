from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from misc.ShowTellModel import ShowTellModel
# from misc.AttentionModel import AttentionModel
# from misc.ShowAttendTellModel import ShowAttendTellModel
# from misc.ShowAttendTellModel_new import ShowAttendTellModel_new
# from misc.TestAttentionModel import TestAttentionModel
from misc.FCModel import FCModel
from misc.Att2inModel import Att2inModel
import torch.nn as nn

def setup(opt):

    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # elif opt.caption_model == 'attention':
    #     return AttentionModel(opt)
    # elif opt.caption_model == 'show_attend_tell':
    #     return ShowAttendTellModel(opt)
    # elif opt.caption_model == 'show_attend_tell_new':
    #     return ShowAttendTellModel_new(opt)
    # elif opt.caption_model == 'test_att':
    #     return TestAttentionModel(opt)
    elif opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model

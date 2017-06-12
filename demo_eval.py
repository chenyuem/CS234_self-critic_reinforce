
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.image as mpimg

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

from misc.resnet_utils import myResnet
import misc.resnet as resnet
import skimage.io
import IPython

from skimage import transform

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='./log_att2in_rl/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--infos_path', type=str, default='./log_att2in_rl/infos_att2in_rl-best.pkl',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='/datadrive/resnet_features_bak/resnet_features/cocotalk_fc',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='/datadrive/resnet_features_bak/resnet_features/cocotalk_att',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='evalscript',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

parser.add_argument('--demo_image', type=str, default='coco_test.jpg')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
ignore = ["id", "batch_size", "beam_size", "start_from"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            if k == 'input_fc_dir' or k == 'input_att_dir':
                continue
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size})
  loader.ix_to_word = infos['vocab']

# fc_feats = np.random.rand(1, 2048)
# att_feats = np.random.rand(1, 14, 14, 2048)
# compute features given a new image
net = getattr(resnet, 'resnet101')()
net.load_state_dict(torch.load('/datadrive/resnet_pretrianed_t7/resnet101.pth'))
my_resnet = myResnet(net)
my_resnet.cuda()
my_resnet.eval()

# filename = 'coco_test.jpg'
import wget
url = opt.demo_image
filename = wget.download(url)
# filename = opt.demo_image

I = skimage.io.imread(filename)
if len(I.shape) == 2:
    I = I[:,:,np.newaxis]
    I = np.concatenate((I,I,I), axis=2)
if I.shape[2] >=3 :
    I = I[:,:,:3]
else:
    print(I.shape)
    raise ValueError('Invalid input image formats')

I = I.astype('float32')/255.0
I = torch.from_numpy(I.transpose([2,0,1])).cuda()
I = Variable(preprocess(I), volatile=True)
tmp_fc, tmp_att = my_resnet(I, 14)

fc_feats = tmp_fc.unsqueeze(0)
att_feats = tmp_att.unsqueeze(0).contiguous()
# IPython.embed()
# tmp = [tmp_fc, tmp_att]
# tmp = [Variable(torch.from_numpy(_), volatile=True).cuda().float() for _ in tmp]
# fc_feats, att_feats = tmp

if 'att2in' in opt.model:
    seq, _, alphas = model.sample_visualize_attention(fc_feats, att_feats, {})
else:
    seq, _ = model.sample(fc_feats, att_feats, {})
sents = utils.decode_sequence(loader.get_vocab(), seq)
print('*'*80)
print(sents)
print('*'*80)

vis_image = mpimg.imread(filename)
if 'att2in' not in opt.model:
    exit()
elif 'att2in_rl' in opt.model:
    # subdir = '/datadrive/att_results/live_demo_att2in_rl/'
    subdir = './live_demo_temp/'
elif 'att2in' in opt.model:
    subdir = '/datadrive/att_results/demo_att2in/'
else:
    assert False
img_name = opt.demo_image.split('/')[-1][:-4]
output_dir = subdir + img_name +'/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# save predicted sentence
fd = open(output_dir + '/' + img_name + '.caption.txt', 'w')
fd.write(sents[0])
fd.close()

for i in range(1, len(alphas)-1):

    a = alphas[i].cpu().data.numpy().squeeze()
    np.unravel_index(a.argmax(), a.shape)

    plt.figure()
    plt.imshow(vis_image)

    a -= np.min(a) * 0.5
    a /= np.max(a)
    attention_map = transform.resize(a, vis_image.shape[:2], order=1)
    plt.imshow(attention_map, alpha=0.3)

    plt.savefig(os.path.join(output_dir, 'demo_image_'+str(i) + '.jpg'))
    plt.close()


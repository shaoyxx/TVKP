import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils.wsad_utils as utils
import numpy as np
from torch.autograd import Variable
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.eval_detection import ANETdetection
import wsad_dataset
from eval.detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
from tensorboard_logger import Logger
import multiprocessing as mp
import options
import model
import proposal_methods as PM
import pandas as pd
from collections import defaultdict
import os
import clip

torch.set_default_tensor_type('torch.cuda.FloatTensor')
@torch.no_grad()
def test(itr, dataset, args, model, clip_model, logger, device, label_token):
    model.eval()
    done = False
    instance_logits_stack = []
    element_logits_stack = []
    labels_stack = []

    back_norms=[]
    front_norms=[]
    ind=0
    proposals = []
    results = defaultdict(dict)
    logits_dict = defaultdict(dict)
    while not done:
        if dataset.currenttestidx % (len(dataset.testidx)//5) ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, labels,vn, done = dataset.load_data(is_training=False)
        label_text = clip_model.encode_text(label_token)
        index = np.argmax(labels, axis=0)
        text_batch = label_text[index]
        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features), label_token, index, is_training=False,seq_len=seq_len)
            element_logits = outputs['cas']
            vnd = vn.decode()
            results[vn] = {'cas':outputs['cas'],'attn':outputs['attn']}
            proposals.append(getattr(PM, args.proposal_method)(vn,outputs))
            logits=element_logits.squeeze(0)
        tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        
        instance_logits_stack.append(tmp)
        labels_stack.append(labels)

    if not os.path.exists('temp'):
        os.mkdir('temp')
    np.save('temp/{}.npy'.format(args.model_name),results)

    instance_logits_stack = np.array(instance_logits_stack)
    labels_stack = np.array(labels_stack)
    proposals = pd.concat(proposals).reset_index(drop=True)

    #CVPR2020
    if 'Thumos14' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95]

        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args,subset='validation')
        dmap_detect.prediction = proposals
        dmap = dmap_detect.evaluate()

    if args.dataset_name == 'Thumos14':
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                labels_stack[i,:] = np.zeros_like(labels_stack[i,:])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print('Classification map %f' %cmap)
    print('||'.join(['map @ {} = {:.3f} '.format(iou[i],dmap[i]*100) for i in range(len(iou))]))
    print('mAP Avg ALL: {:.3f}'.format(sum(dmap)/len(iou)*100))
    
    logger.log_value('Test Classification mAP', cmap, itr)
    for item in list(zip(dmap,iou)):
    	logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)
    utils.write_to_file(args.dataset_name, dmap * 100, itr)
    return iou,dmap

if __name__ == '__main__':
    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset,args.dataset)(args)

    clip_model, _ = clip.load('/dataset/ViT-B-32.pt', jit=False)
    clip_model.eval()
    clip_model = clip_model.to(device)

    for param in clip_model.parameters():
        param.requires_grad = False

    classes = ['baseball pitch', 'basketball dunk', 'billiards', 'clean jerk', 'cliff diving', 'cricket bowling',
               'cricket shot', 'diving', 'frisbee catch', 'golf swing', 'hammer throw', 'high jump', 'javelin throw',
               'long jump', 'pole vault', 'shot put', 'soccer penalty', 'tennis swing', 'throw discus', 'volleyball spiking', 'background']

    label_prompts = [f"{c}" for c in classes]
    label_token = clip.tokenize(label_prompts).to(device)

    model = getattr(model, args.use_model)(dataset.feature_size, clip_model, classes, opt=args).to(device)

    model.load_state_dict(torch.load('./ckpt/best_TFEDCN_1.pkl'))
    logger = Logger('./logs/test_' + args.model_name)
    pool = mp.Pool(5)

    iou,dmap = test(-1, dataset, args, model, clip_model, logger, device, label_token)
    print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(dmap[:5])*100,np.mean(dmap[:7])*100,np.mean(dmap)*100))

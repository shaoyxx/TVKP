import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
import numpy as np
from tensorboard_logger import Logger

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(itr, dataset, args, model, clip_model, optimizer, logger, device, label_token):
    model.train()
    label_text = clip_model.encode_text(label_token)
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]

    index = np.argmax(labels, axis=1)
    text_batch = label_text[index]

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    #interative
    pseudo_label = None

    outputs = model(features, label_token, index, seq_len=seq_len,is_training=True,itr=itr,opt=args)
    total_loss = model.criterion(outputs, labels, seq_len=seq_len,device=device,logger=logger,opt=args,itr=itr,pairs_id=pairs_id,inputs=features)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import copy
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from decoder import ContextDecoder
_tokenizer = _Tokenizer()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class NCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, features, weight=None, tem=None):
        batch_size = features.size(0) / 2
        device = features.device
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        if weight == None:
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        else:
            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) * weight
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits * tem
        loss = self.criterion(logits, labels)
        return loss

class Modality_Enhancement_Module(torch.nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()
        embed_dim = 1024
        dropout_ratio = args['opt'].dropout_ratio
        self.channel_conv = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.6))

        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 256, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(dropout_ratio),
                                       nn.Conv1d(256, 256, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(256, 1, 1),
                                       nn.Dropout(dropout_ratio),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)

    def forward(self,vfeat,ffeat):
        channel_attn = self.channel_conv(self.channel_avg(vfeat))
        bit_wise_attn = self.channel_conv(self.channel_avg(ffeat))

        filter_feat = torch.sigmoid(channel_attn) * torch.sigmoid(bit_wise_attn) * vfeat

        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat

class TFE_DC_Module(nn.Module):
    def __init__(self, n_feature, args):
        super().__init__()
        dropout_ratio = args['opt'].dropout_ratio
        embed_dim = 1024
        self.layer1 = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=2 ** 0, dilation=2 ** 0),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.6))
        self.layer2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 1, dilation=2 ** 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.6))
        self.layer3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 2, dilation=2 ** 2),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.6))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 256, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(dropout_ratio),
                                       nn.Conv1d(256, 256, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(256, 1, 1),
                                       nn.Dropout(dropout_ratio),
                                       nn.Sigmoid())

    def forward(self, x):
        out_1 = self.layer1(x)
        out_feature_1 = torch.sigmoid(out_1) * x
        out_attention1 = self.attention(out_feature_1)

        out_2 = self.layer2(out_1)
        out_feature_2 = torch.sigmoid(out_2) * x
        out_attention2 = self.attention(out_feature_2)

        out_3 = self.layer3(out_2)
        out_feature = torch.sigmoid(out_3) * x
        out_attention3 = self.attention(out_feature)

        out_attention = (out_attention1 + out_attention2 + out_attention3) / 3.0

        return out_attention, out_feature

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        dtype = clip_model.dtype
        ctx_dim = 512

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class TFEDCN(torch.nn.Module):
    def __init__(self, n_feature, clip_model, classnames, **args):
        super().__init__()

        dropout_ratio = args['opt'].dropout_ratio
        self.vAttn = getattr(model,args['opt'].AWM)(1024,args)
        self.fAttn = getattr(model,args['opt'].TCN)(1024,args)

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio)
        )

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        self.nce = NCELoss()

        self.apply(weights_init)

        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.pdl = ContextDecoder(transformer_width=512,
                                  transformer_heads=8,
                                  transformer_layers=1,
                                  visual_dim=512,
                                  dropout=dropout_ratio)

    def forward(self, inputs, label_token, index, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n = feat.size()
        num = int(n // 7)

        f_atn,ffeat = self.fAttn(feat[:,1024:,:])
        v_atn,vfeat = self.vAttn(feat[:,:1024,:],ffeat)
        x_atn = f_atn * 0.5 + v_atn * 0.5
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = self.fusion(nfeat).transpose(-1, -2)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        label_text = self.text_encoder(prompts, tokenized_prompts)

        _, sorted_idx = torch.sort(f_atn.transpose(-1, -2), dim=1, descending=True)
        feat_idx = sorted_idx[:, :num, :].expand([-1, -1, 512])
        snip_feat = torch.gather(nfeat, 1, feat_idx).view(b * num, -1)

        back_idx = sorted_idx[:, num:num*2, :].expand([-1, -1, 512])
        back_feat = torch.gather(nfeat, 1, back_idx).view(b * num, -1)

        snip_idx = sorted_idx[:, :20, :].expand([-1, -1, 512])
        global_feat = torch.gather(nfeat, 1, snip_idx)

        classifier, mu, log_var = self.pdl(label_text.unsqueeze(0).repeat(b, 1, 1), global_feat)

        nfeat_norm = nfeat / nfeat.norm(dim=-1, keepdim=True)
        classifier_norm = classifier / classifier.norm(dim=-1, keepdim=True)
        x_cls = torch.mean(torch.einsum('bnd,bkcd->bknc', nfeat_norm, classifier_norm) * 50, dim=1)

        text_batch = label_text[index]
        text_batch = text_batch.unsqueeze(1).repeat(1, num, 1).view(b * num, -1)

        return {'feat': nfeat, 'cas': x_cls, 'attn': x_atn.transpose(-1, -2), 'v_atn': v_atn.transpose(-1, -2),
                'f_atn': f_atn.transpose(-1, -2), 'feat_proj': snip_feat, 'text_proj': text_batch,
                'back_proj': back_feat}


    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']
        f_proj = outputs['feat_proj']
        t_proj = outputs['text_proj']
        b_proj = outputs['back_proj']

        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, f_atn, include_min=True)

        loss_mil_orig = self.topkloss(element_logits,
                                      labels,
                                      is_back=True,
                                      rat=args['opt'].k,
                                      reduce=None)
        # SAL
        loss_mil_supp = self.topkloss(element_logits_supp,
                                      labels,
                                      is_back=False,
                                      rat=args['opt'].k,
                                      reduce=None)

        v_loss_norm = v_atn.abs().mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.abs().mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        nce_loss = (self.nce(torch.cat([f_proj, t_proj], dim=0), tem=5) + self.nce(torch.cat([t_proj, f_proj], dim=0), tem=5)) / 2
        nce_loss_2 = (self.nce(torch.cat([f_proj, b_proj], dim=0), tem=5) + self.nce(torch.cat([b_proj, f_proj], dim=0), tem=5)) / 2

        # total loss
        total_loss = (loss_mil_orig.mean() +
                      loss_mil_supp.mean()
                      + args['opt'].alpha1 * (f_loss_norm + v_loss_norm)
                      + args['opt'].alpha2 * f_loss_guide
                      + args['opt'].alpha3 * v_loss_guide
                      + args['opt'].alpha4 * mutual_loss + nce_loss * 0.1 + nce_loss_2 * 0.1
                      )

        return total_loss

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)

        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)

        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )

        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))

        if reduce is not None:
            milloss = milloss.mean()

        return milloss

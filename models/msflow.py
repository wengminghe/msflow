import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np

from .extractors import build_extractor
from .flow_models import build_msflow_model


class MSFlow(nn.Module):
    def __init__(self, extractor, c_conds, parallel_blocks, pool_type='avg', clamp_alpha=1.9, **kwargs):
        super().__init__()
        self.c_conds = c_conds

        self.extractor = build_extractor(extractor, pretrained=True)
        self.parallel_flows, self.fusion_flow = build_msflow_model(self.extractor.out_channels, c_conds, parallel_blocks, clamp_alpha)

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(3, 2, 1)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(3, 2, 1)
        else:
            self.pool = nn.Identity()

        for param in self.extractor.parameters():
            param.requires_grad = False
        self.extractor.eval()

    def model_forward(self, image):
        h_list = self.extractor(image)

        z_list = []
        parallel_jac_list = []
        for h, parallel_flow, c_cond in zip(h_list, self.parallel_flows, self.c_conds):
            y = self.pool(h)
            B, _, H, W = y.shape
            cond = self.positionalencoding2d(c_cond, H, W).to(h.device).unsqueeze(0).repeat(B, 1, 1, 1)
            z, jac = parallel_flow(y, [cond, ])
            z_list.append(z)
            parallel_jac_list.append(jac)

        z_list, fuse_jac = self.fusion_flow(z_list)
        jac = fuse_jac + sum(parallel_jac_list)

        return z_list, jac

    def forward(self, image):
        z_list, jac = self.model_forward(image)
        outputs = {
            'z_list': z_list,
            'jac': jac
        }

        return outputs

    @staticmethod
    def postprocess(size_list, outputs_list, input_size, top_k=0.03):
        logp_maps = [list() for _ in size_list]
        prop_maps = [list() for _ in size_list]
        for l, outputs in enumerate(outputs_list):
            outputs = torch.cat(outputs, 0)
            logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                                         size=input_size, mode='bilinear', align_corners=True).squeeze(1)
            output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
            prob_map = torch.exp(output_norm)  # convert to probs in range [0:1]
            prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                                         size=input_size, mode='bilinear', align_corners=True).squeeze(1)

        logp_map = sum(logp_maps)
        logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prop_map_mul = torch.exp(logp_map)
        anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
        batch = anomaly_score_map_mul.shape[0]
        top_k = int(input_size[0] * input_size[1] * top_k)
        anomaly_score = np.mean(
            anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
            axis=1)

        prop_map_add = sum(prop_maps)
        prop_map_add = prop_map_add.detach().cpu().numpy()
        anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

        return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()

    @torch.no_grad()
    def predict(self, image, top_k=0.03):
        z_list, jac = self.model_forward(image)
        outputs_list = [list() for _ in self.parallel_flows]
        size_list = []
        for lvl, z in enumerate(z_list):
            size_list.append(list(z.shape[-2:]))
            logp = - 0.5 * torch.mean(z ** 2, 1)
            outputs_list[lvl].append(logp)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = self.postprocess(size_list, outputs_list, image.shape[-2:], top_k=top_k)
        return anomaly_score, anomaly_score_map_add

    @staticmethod
    def positionalencoding2d(D, H, W):
        """
        :param D: dimension of the model
        :param H: H of the positions
        :param W: W of the positions
        :return: DxHxW position matrix
        """
        if D % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
        P = torch.zeros(D, H, W)
        # Each dimension use half of D
        D = D // 2
        div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
        pos_w = torch.arange(0.0, W).unsqueeze(1)
        pos_h = torch.arange(0.0, H).unsqueeze(1)
        P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        return P



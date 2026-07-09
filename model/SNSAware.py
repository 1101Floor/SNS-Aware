"""Cleaned SNS-Aware model definition.
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .dist import Normal
from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from .mlp import MLP as TrajectoryMLP
from utils.torchutils import *
from utils.utils import initialize_weights
from .posenc import PositionalAgentEncoding

class HistoryEncoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=4):
        super().__init__()

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.he_tf_layer

        self.agent_enc_shuffle = pos_enc['agent_enc_shuffle']

        self.pooling = args.pooling

        self.in_dim = in_dim
        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = TransformerEncoder(encoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, traj_in, agent_mask, agent_enc_shuffle=None):

        agent_num = traj_in.shape[1]

        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.obs_len, agent_num, 1, self.model_dim])

        src_agent_mask = agent_mask.clone()

        history_enc = self.tf_encoder(tf_in_pos, mask=src_agent_mask, num_agent=agent_num)
        history_rs = history_enc.view(-1, agent_num, self.model_dim)
        if self.pooling == 'mean':
            agent_history = torch.mean(history_rs, dim=0)  
        else:
            agent_history = torch.max(history_rs, dim=0)[0]
        return history_enc, agent_history

class FutureEncoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=4):
        super().__init__()

        self.pred_len = args.pred_len

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.fe_tf_layer

        self.pooling = args.pooling
        self.cross_motion_only = args.cross_motion_only

        self.in_dim = in_dim

        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        decoder_layers = TransformerDecoderLayer(self.model_dim, self.nhead, self.ff_dim,
                                                 self.dropout, cross_motion_only=self.cross_motion_only)
        self.tf_decoder = TransformerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, traj_in, history_enc, agent_mask, agent_enc_shuffle=None):

        agent_num = traj_in.shape[1]

        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.pred_len, agent_num, 1, self.model_dim])

        mem_agent_mask = agent_mask.clone()
        tgt_agent_mask = agent_mask.clone()

        tf_out, _ = self.tf_decoder(tf_in_pos, history_enc, memory_mask=mem_agent_mask,
                                    tgt_mask=tgt_agent_mask, num_agent=agent_num)
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        # [N d_model-256]
        if self.pooling == 'mean':
            agent_future = torch.mean(tf_out, dim=0)
        else:
            agent_future = torch.max(tf_out, dim=0)[0]

        return agent_future

class FutureDecoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=2):
        super().__init__()

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.pred_dim = args.pred_dim

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.fd_tf_layer

        self.cross_motion_only = args.cross_motion_only

        self.in_dim = in_dim + args.nz  
        self.out_mlp_dim = args.fd_out_mlp_dim

        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        decoder_layers = TransformerDecoderLayer(self.model_dim, self.nhead, self.ff_dim,
                                                 self.dropout, cross_motion_only=self.cross_motion_only)
        self.tf_decoder = TransformerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, dec_in, z, sample_num, agent_num, agent_mask,
                history_enc, agent_enc_shuffle=None, need_weights=False):

        z_in = z.unsqueeze(0).repeat_interleave(self.pred_len, dim=0)  
        z_in = z_in.view(self.pred_len, agent_num, sample_num, z.shape[-1])  

        in_arr = [dec_in, z_in]

        dec_in_z = torch.cat(in_arr, dim=-1).reshape([agent_num * 12, sample_num, -1])

        tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.pred_len, agent_num, sample_num, self.model_dim])

        mem_agent_mask = agent_mask.clone()
        tgt_agent_mask = agent_mask.clone()

        tf_out, attn_weights = self.tf_decoder(tf_in_pos, history_enc, memory_mask=mem_agent_mask, tgt_mask=tgt_agent_mask,
                                               seq_mask=True, num_agent=agent_num, need_weights=need_weights)

        return tf_out, attn_weights

    


class PatchEmbedding(nn.Module):
    """Layer that divides images into patches and embeds them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class ViTFeedForward(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super(ViTFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTFeedForward(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PyTorchViT(nn.Module):
    """PyTorch native Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, num_layers=12, num_heads=12, mlp_dim=3072, 
                 num_classes=1000, dropout=0.1):
        super(PyTorchViT, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_features=False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks with feature extraction
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        x = self.ln(x)
        
        if return_features:
            return x, features  # Return features from all layers
        else:
            return x

class ViTSceneEncoder(nn.Module):
    """
    Minimal ViT scene encoder that reuses the original PyTorchViT modules.
    Input:  scene_map  [B,H,W] or [B,1,H,W]  (grayscale, float 0~1)
    Output: scene_feat [B, embedding_dim]
    """
    def __init__(self, embedding_dim=64, img_size=224, patch_size=16,
                 embed_dim=768, num_layers=6, num_heads=8,
                 freeze_vit=False):
        super().__init__()

        self.img_size = img_size
        self.embedding_dim = embedding_dim

        # --- reuse original PyTorchViT ---
        self.vit = PyTorchViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,          # vit expects RGB
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=embed_dim * 4,
            num_classes=1000,
            dropout=0.1
        )

        # freeze if needed
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # project CLS token to embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embedding_dim),
            nn.ReLU()
        )
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        # ImageNet norm (same as before, but inline use)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, scene_map):
        """
        scene_map: [B,H,W] or [B,1,H,W] or [B,3,H,W]
        """
        B = scene_map.size(0)

        # ---- 1) make shape [B,1,H,W] ----
        if scene_map.dim() == 3:
            scene_map = scene_map.unsqueeze(1)

        # ---- 2) grayscale -> 3-channel RGB ----
        if scene_map.size(1) == 1:
            scene_map = scene_map.expand(B, 3, scene_map.size(2), scene_map.size(3))

        # ---- 3) resize to vit input ----
        if scene_map.size(2) != self.img_size or scene_map.size(3) != self.img_size:
            scene_map = F.interpolate(scene_map, size=(self.img_size, self.img_size),
                                      mode="bilinear", align_corners=False)

        # ---- 4) normalize (assume input in [0,1]) ----
        #cene_map = (scene_map - self.mean) / self.std

        # ---- 5) vit forward ----
        # vit output: [B, 1+num_patches, embed_dim]
        scene_map = (scene_map - self.mean) / self.std
        features = self.vit(scene_map)
        
        patch_features = features[:, 1:, :]
        pooled_features, attention_weights = self.attention_pool(
            patch_features, patch_features, patch_features
        )
        global_features = pooled_features.mean(dim=1)
        #print("vit_tokens",vit_tokens.shape)
        #cls_token = vit_tokens[:, 0]      # [B, embed_dim]
        #print("cls_token",cls_token.shape)
        # ---- 6) project to scene embedding ----
        scene_feat = self.proj(global_features) # [B, embedding_dim]
        #print(scene_feat.shape,scene_feat)
        return scene_feat
            
class SceneModel(nn.Module):
    def __init__(self,
                 traj_feat_dim=2023,
                 hidden_dim=256,
                 scene_emb_dim=16,  # ViTSceneEncoder 输出维度
                 img_size=224):
        super().__init__()

        # ① 场景编码器
        self.scene_encoder = ViTSceneEncoder(
            embedding_dim=scene_emb_dim,
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            num_layers=6,
            num_heads=8,
            freeze_vit=False       # False = 参与训练
        )

    def forward(self, scene_map):
        """
        x_traj:   (B, 1024, 2023)
        scene_map:(B, 1, H, W) or (B, H, W)
        """

        # --- A) 场景特征 ---
        # scene_feat: (B, scene_emb_dim)
        scene_feat = self.scene_encoder(scene_map)

        return scene_feat

class GroupGenerator(nn.Module):
    def __init__(self, d_type='learned', th=6000.0, in_channels=16, hid_channels=16, n_head=1, dropout=0):
        super().__init__()
        self.d_type = d_type
        self.init_th = th
        if d_type == 'learned':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 1),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(hid_channels),
                                           nn.Dropout(dropout, inplace=True),
                                           nn.Conv2d(hid_channels, n_head, 1),)
        elif d_type == 'estimate_th':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, n_head, 1),)
        elif d_type == 'learned_l2norm':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0)))
        #self.th = th if type(th) == float else nn.Parameter(torch.Tensor([1]))
        self.th = nn.Parameter(torch.tensor([float(th)], dtype=torch.float32))
    def find_group_indices(self, v, dist_mat, th=None):
        if th is None:
            th = self.th
        #rint('find_group_indices',th)
        n_ped = v.size(-1)
        mask = torch.ones_like(dist_mat).mul(1e4).triu()

        top_row, top_column = torch.nonzero(
            dist_mat.tril(diagonal=-1).add(mask).le(th),
            as_tuple=True
        )

        indices_raw = torch.arange(n_ped, dtype=top_row.dtype, device=v.device)

        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c

        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_row.dtype, device=v.device)

        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j

        return indices

    def find_group_indices_ratio(self, v, dist_mat):
        n_ped = v.size(-1)
        group_num = n_ped - (n_ped + self.th - 1) // self.th
        top_list = (1. / dist_mat).tril(diagonal=-1).view(-1).topk(k=group_num)[1]
        top_row, top_column = top_list // n_ped, top_list % n_ped
        indices_raw = torch.arange(n_ped, dtype=top_list.dtype, device=v.device)
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_list.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
        return indices

    def group_backprop_trick_threshold(self, v, dist_mat, tau=1, hard=False, th=None):
        if th is None:
            th = self.th
        #print(th)
        #rint(th, dist_mat)
        sig = (-(dist_mat - th) / tau).sigmoid()
        sig_norm = sig / sig.sum(dim=0, keepdim=True)
        v_soft = v @ sig_norm

        return (v - v_soft).detach() + v_soft if hard else v_soft

    def forward(self, v, v_abs, tau=0.1, hard=True):
        assert v.size(0) == 1
        n_ped = v.size(-1)
        th = self.th

        # Measure similarity between agent pairs
        if self.d_type == 'euclidean':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned_l2norm':
            #print('v_abs',v_abs)
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            #print('temp',temp)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)  # symmetric
        elif self.d_type == 'estimate_th':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = temp - temp.transpose(-2, -1)
            dist_mat = temp.norm(p=2, dim=1)
            temp_for_cnn = temp.mean(dim=2)
            raw_th = self.group_cnn(temp_for_cnn).mean().exp()
            th = self.th * raw_th
        else:
            raise NotImplementedError

        dist_mat = dist_mat.squeeze(dim=0).mean(dim=0)
        indices = self.find_group_indices(v, dist_mat, th=th)
        v = self.group_backprop_trick_threshold(v, dist_mat, tau=tau, hard=hard, th=th)
        return v, indices

    @staticmethod
    def ped_group_pool(v, indices):
        assert v.size(-1) == indices.size(0)
        n_ped = v.size(-1)
        n_ped_pool = indices.unique().size(0)
        v_pool = torch.zeros(v.shape[:-1] + (n_ped_pool,), device=v.device)
        v_pool.index_add_(-1, indices, v)
        v_pool_num = torch.zeros((v.size(0), 1, 1, n_ped_pool), device=v.device)
        v_pool_num.index_add_(-1, indices, torch.ones((v.size(0), 1, 1, n_ped), device=v.device))
        v_pool /= v_pool_num
        return v_pool

    @staticmethod
    def ped_group_unpool(v, indices):
        assert v.size(-1) == indices.unique().size(0)
        return torch.index_select(input=v, dim=-1, index=indices)

    @staticmethod
    def ped_group_mask(indices):
        mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
        for i in indices.unique():
            idx_list = torch.nonzero(indices.eq(i))
            for idx in idx_list:
                mask[idx, idx_list] = 1
        return mask


class GroupIntegrator(nn.Module):
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, pred_seq_len=12):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        if mix_type == 'mlp':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * pred_seq_len * n_mix, out_channels * pred_seq_len,
                                                       kernel_size=1), )
        elif mix_type == 'cnn':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * n_mix, out_channels,
                                                       kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, v_stack):
        n_batch, n_ped = v_stack[0].shape[0], v_stack[0].shape[3]
        if self.mix_type == 'sum':
            v = torch.stack(v_stack, dim=0).sum(dim=0)
        elif self.mix_type == 'mean':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
        elif self.mix_type == 'mlp':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v_stack = torch.cat(v_stack, dim=1).reshape(n_batch, -1, 1, n_ped)
            v = v + self.st_gcns_mix(v_stack).view(n_batch, -1, self.pred_seq_len, n_ped)
        elif self.mix_type == 'cnn':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v = v + self.st_gcns_mix(torch.cat(v_stack, dim=1))
        else:
            raise NotImplementedError
        return v




def generate_identity_matrix(v):
    i = [torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1),
         torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1)]
    return i


class GPGraph(nn.Module):
    def __init__(self, baseline_model, in_channels=2, out_channels=5, obs_seq_len=8, pred_seq_len=12,
                 d_type='learned_l2norm', d_th='learned', mix_type='mlp', group_type=None, weight_share=True):
        super().__init__()

        self.baseline_model = baseline_model
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.mix_type = mix_type
        self.weight_share = weight_share

        group_type = (True,) * 3 if group_type is None else group_type
        self.include_original = group_type[0]
        self.include_inter_group = group_type[1]
        self.include_intra_group = group_type[2]

        self.group_gen = GroupGenerator(d_type=d_type, th=d_th, in_channels=in_channels, hid_channels=8)
        self.group_mix = GroupIntegrator(mix_type=mix_type, n_mix=sum(group_type),
                                         out_channels=out_channels, pred_seq_len=pred_seq_len)

    def forward(self, v_abs, v_rel):
        v_stack = []
        # Agent graph
        if self.include_original:
            # Agent-agent interaction
            v = v_rel
            i = generate_identity_matrix(v)
            v = v.permute(0, 2, 3, 1)
            v = self.baseline_model(v, i) if self.weight_share else self.baseline_model[0](v, i)
            v = v.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v)
        # Intra-/Inter-group graph
        v_rel, indices = self.group_gen(v_rel, v_abs, hard=True)

        if self.include_inter_group:
            # Inter-group interaction
            v_e = self.group_gen.ped_group_pool(v_rel, indices)  # Agent Group Pooling
            i_e = generate_identity_matrix(v_e)
            v_e = v_e.permute(0, 2, 3, 1)
            v_e = self.baseline_model(v_e, i_e) if self.weight_share else self.baseline_model[1](v_e, i_e)
            v_e = v_e.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_e = self.group_gen.ped_group_unpool(v_e, indices)  # Agent Group Unpooling
            v_stack.append(v_e)
        if self.include_intra_group:
            # Intra-group interaction
            v_i = v_rel
            mask = self.group_gen.ped_group_mask(indices)
            i_i = generate_identity_matrix(v_i)
            v_i = v_i.permute(0, 2, 3, 1)
            #print("v_3",v_i.shape, i_i)
            v_i = self.baseline_model(v_i, i_i, mask) if self.weight_share else self.baseline_model[2](v_i, i_i, mask)
            v_i = v_i.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v_i)
        # Group Integration
        v = self.group_mix(v_stack)
        return v, indices

    
class SNSAware(nn.Module):
    def __init__(self, args, baseline_model):
        super().__init__()

        self.device = torch.device('cpu')

        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.kernel_size_gcn = 3
        self.input_feat = 2
        self.output_feat = 2
        self.n_stgcnn = 1
        self.scene_emb_dim = args.scene_emb_dim
        self.feature_up = 16
        # position encoding
        self.pos_enc = {
            'pos_concat': args.pos_concat,
            'max_agent_len': 128,  # 128
            'use_agent_enc': False,  # False
            'agent_enc_learn': False,  # False
            'agent_enc_shuffle': False  # False
        }
        self.max_train_agent = args.max_train_agent
        self.rand_rot_scene = args.rand_rot_scene
        self.discrete_rot = args.discrete_rot

        self.model_dim = args.tf_model_dim
        self.pred_dim = args.pred_dim
        
        self.compute_sample = True

        num_dist_params = 2 * args.nz

        self.he_out_mlp_dim = args.he_out_mlp_dim
        if self.he_out_mlp_dim is None:
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.he_out_mlp = TrajectoryMLP(self.model_dim, args.he_out_mlp_dim, 'relu')
            self.p_z_net = nn.Linear(self.he_out_mlp.out_dim, num_dist_params)
        initialize_weights(self.p_z_net.modules())


        self.fe_out_mlp_dim = args.fe_out_mlp_dim
        if self.fe_out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.fe_out_mlp = TrajectoryMLP(self.model_dim, args.fe_out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.fe_out_mlp.out_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())


        self.fd_out_mlp_dim = args.fd_out_mlp_dim
        if self.fd_out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, self.pred_dim)
        else:
            self.fd_out_mlp = TrajectoryMLP(self.model_dim, args.fd_out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.fd_out_mlp.out_dim, self.pred_dim)
        initialize_weights(self.out_fc.modules())

        # models
        self.history_encoder = HistoryEncoder(args, self.pos_enc)
        self.future_encoder = FutureEncoder(args, self.pos_enc)
        self.future_decoder = FutureDecoder(args, self.pos_enc)
        
        ########
        self.num_tcn_layers_feature_up = args.num_tcn_layers
        self.tcn_layers_feature_up = nn.ModuleList()
        self.tcn_layers_feature_up.append(nn.Sequential(
            nn.Conv1d(self.input_feat, self.feature_up, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.num_tcn_layers_feature_up):
            self.tcn_layers_feature_up.append(nn.Sequential(
                nn.Conv1d(self.feature_up, self.feature_up, 3, padding=1),
                nn.PReLU()
            ))
            
        ########
        self.num_tcn_layers_fusion = args.num_tcn_layers
        self.tcn_layers_fusion = nn.ModuleList()
        self.tcn_layers_fusion.append(nn.Sequential(
            nn.Conv1d(self.input_feat + self.scene_emb_dim + self.feature_up, self.input_feat, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.num_tcn_layers_fusion):
            self.tcn_layers_fusion.append(nn.Sequential(
                nn.Conv1d(self.input_feat, self.input_feat, 3, padding=1),
                nn.PReLU()
            ))
        ########
        self.num_tcn_layers = args.num_tcn_layers
        self.tcn_layers = nn.ModuleList()
        self.tcn_layers.append(nn.Sequential(
            nn.Conv1d(self.obs_len, self.pred_len, 3, padding=1),
            nn.PReLU()
        ))
        for j in range(1, self.num_tcn_layers):
            self.tcn_layers.append(nn.Sequential(
                nn.Conv1d(self.pred_len, self.pred_len, 3, padding=1),
                nn.PReLU()
            ))
            
        ########    
        self.param_annealers = nn.ModuleList()
        self.SE = SceneModel(traj_feat_dim=2023,
                 hidden_dim=256,
                 scene_emb_dim=self.scene_emb_dim,  # ViTSceneEncoder 输出维度
                 img_size=224)
        
        self.fus_fc_1 = nn.Linear(128 + self.scene_emb_dim, 128)
        
        self.IE = GPGraph(baseline_model, in_channels=self.input_feat, out_channels=self.output_feat, obs_seq_len=self.obs_len, 
                     pred_seq_len=args.pred_len, d_type='estimate_th', d_th=6000, mix_type='mlp',group_type=(True, True, True), 
                           weight_share=True)
        self.fus_fc_3 = nn.Linear(self.input_feat*2, self.output_feat)
    def set_device(self, device):
        self.device = device
        self.to(device)

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def set_data(self, pre_motion, fut_motion, pre_motion_mask, fut_motion_mask, v ,a_dis, a_tpca, a_s, \
                 reference_image, v_image_obs, V_obs_abs_1, V_obs_tmp_1):
        device = self.device

        fut_motion_orig = fut_motion.transpose(1, 2)  

        pre_motion = pre_motion.permute(2, 0, 1)
        fut_motion = fut_motion.permute(2, 0, 1)

        if self.training and pre_motion.shape[1] > self.max_train_agent:
            ind = np.random.choice(pre_motion.shape[1], self.max_train_agent).tolist()
            ind = torch.tensor(ind).to(device)

            pre_motion = torch.index_select(pre_motion, 1, ind).contiguous()
            pre_motion_mask = torch.index_select(pre_motion_mask, 0, ind).contiguous()  
            fut_motion = torch.index_select(fut_motion, 1, ind).contiguous()
            fut_motion_mask = torch.index_select(fut_motion_mask, 0, ind).contiguous()  
            fut_motion_orig = torch.index_select(fut_motion_orig, 0, ind).contiguous()

        self.agent_num = pre_motion.shape[1]

        self.scene_orig = pre_motion[[-1]].view(-1, 2).mean(dim=0)  

        # rotate the scene
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=20, size=(1,)).to(device) * (np.pi / 10)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2  
            pre_motion, pre_motion_scene_norm = rotation_2d_torch(pre_motion, theta, self.scene_orig)
            fut_motion, fut_motion_scene_norm = rotation_2d_torch(fut_motion, theta, self.scene_orig)
            fut_motion_orig, fut_motion_orig_scene_norm = rotation_2d_torch(fut_motion_orig, theta, self.scene_orig)
        else:
            theta = torch.zeros(1).to(device)
            pre_motion_scene_norm = pre_motion - self.scene_orig
            fut_motion_scene_norm = fut_motion - self.scene_orig

        pre_vel = pre_motion[1:] - pre_motion[:-1, :]
        pre_vel = torch.cat([pre_vel[[0]], pre_vel], dim=0)
        fut_vel = fut_motion - torch.cat([pre_motion[[-1]], fut_motion[:-1, :]])

        cur_motion = pre_motion[[-1]][0]
        mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        agent_mask = mask  # [N N]

        # assign values
        self.agent_mask = agent_mask

        self.pre_motion_mask = pre_motion_mask
        self.fut_motion_mask = fut_motion_mask

        self.pre_motion = pre_motion
        self.pre_vel = pre_vel
        self.pre_motion_scene_norm = pre_motion_scene_norm

        self.fut_motion = fut_motion
        self.fut_vel = fut_vel
        self.fut_motion_scene_norm = fut_motion_scene_norm

        self.fut_motion_orig = fut_motion_orig
        self.v = v
        self.a_dis = a_dis
        self.a_tpca = a_tpca
        self.a_s = a_s
        self.reference_image = reference_image 
        self.v_image_obs = v_image_obs
        self.V_obs_abs_1 = V_obs_abs_1 
        self.V_obs_tmp_1 = V_obs_tmp_1

    def temporal_convolution(self, pre_motion_scene_norm):
        dec_in = pre_motion_scene_norm.transpose(0, 1) 
        dec_in = self.tcn_layers[0](dec_in)
        for k in range(1, self.num_tcn_layers):
            dec_in = self.tcn_layers[k](dec_in) + dec_in

        dec_in = dec_in.transpose(0, 1).unsqueeze(2)  
        return dec_in

    def encode_history(self):
        he_in = torch.cat([self.pre_motion_scene_norm, self.pre_vel], dim=-1)
        history_enc, agent_history = self.history_encoder(he_in, self.agent_mask)
        return history_enc, agent_history

    def encode_future(self, history_enc):
        fe_in = torch.cat([self.fut_motion_scene_norm, self.fut_vel], dim=-1)
        agent_future = self.future_encoder(fe_in, history_enc, self.agent_mask)
        return agent_future

    def decode_future(self, z, sample_num, history_enc, dec_in=None, need_weights=False):
        if dec_in is None:
            dec_in = self.temporal_convolution(self.pre_motion_scene_norm)
        else:
            assert len(dec_in.shape) == 4 
        assert z.shape[0] == self.agent_num * sample_num 
        dec_in = dec_in.repeat_interleave(sample_num, dim=2) 
        history_enc = history_enc.repeat_interleave(sample_num, dim=2)  

        dec_motion, attn_weights = self.future_decoder(dec_in, z, sample_num, self.agent_num,
                                                       self.agent_mask, history_enc, need_weights=need_weights)

        out_tmp = dec_motion.view(-1, dec_motion.shape[-1])  
        if self.fd_out_mlp_dim is not None: 
            cat_arr = [out_tmp]
            out_tmp = torch.cat(cat_arr, dim=-1)
            out_tmp = self.fd_out_mlp(out_tmp)
        seq_out = self.out_fc(out_tmp).view(dec_motion.shape[0], -1, self.pred_dim)  
        seq_out = seq_out.view(-1, self.agent_num * sample_num, seq_out.shape[-1])
        dec_motion = seq_out + self.scene_orig
        dec_motion = dec_motion.transpose(0, 1).contiguous() 
        if sample_num > 1:
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:]) 
        return dec_motion, attn_weights

    def get_prior(self, agent_history, sample_num=1):
        if self.he_out_mlp_dim is not None:  
            agent_history = self.he_out_mlp(agent_history)
        h = agent_history.repeat_interleave(sample_num, dim=0)
        p_z_params = self.p_z_net(h)
        p_z_dist = Normal(params=p_z_params)
        return p_z_dist

    def get_posterior(self, agent_future):
        if self.fe_out_mlp_dim is not None: 
            agent_future = self.fe_out_mlp(agent_future)
        q_z_params = self.q_z_net(agent_future)
        q_z_dist = Normal(params=q_z_params)
        return q_z_dist

    def inference(self, sample_num, need_weights=False):
        history_enc, agent_history = self.encode_history()
        p_z_dist = self.get_prior(agent_history, sample_num=sample_num)
        z = p_z_dist.sample()
        dec_motion, attn_weights = self.decode_future(z, sample_num, history_enc, need_weights=need_weights)
        if sample_num == 1:
            dec_motion = dec_motion.unsqueeze(1)
        return dec_motion, attn_weights 

    def forward(self, sample_num):
        #Intraction Extractor
        MLI, _ = self.IE(self.V_obs_abs_1, self.V_obs_tmp_1)#[1, 2, 12, 6]
        MLI = MLI.squeeze(0)#[2, 12, 6]
        MLI = MLI.transpose(0, 1) #[12, 2, 6]
        
        SF = self.SE(self.reference_image)#[1,4]
        SF = SF.unsqueeze(-1).expand(MLI.shape[0], -1, MLI.shape[2])#[12, 4, 6]
        
        ####
        BF = self.temporal_convolution(self.pre_motion_scene_norm)#[12, 6, 1, 2]
        BF = BF.squeeze(2)#[12, 6, 2]
        BF = BF.permute(0, 2, 1)#[12, 2，6]
        
        ####
        BF_feature_up = self.tcn_layers_feature_up[0](BF)
        for k in range(1, self.num_tcn_layers_feature_up):
            BF_feature_up = self.tcn_layers_feature_up[k](BF_feature_up) + BF_feature_up #[12, 16, 6]
         
        ####    
        Fusion = torch.cat([MLI, BF_feature_up, SF], dim = 1)#[12, 22, 6]
        
        ####
        dec_in = self.tcn_layers_fusion[0](Fusion)
        for k in range(1, self.num_tcn_layers_fusion):
            dec_in = self.tcn_layers_fusion[k](dec_in) + dec_in #[12, 2, 6]
        
        dec_in = dec_in.permute(0, 2, 1).unsqueeze(2)  #[12, 6, 1, 2]
        
        ####
        history_enc, agent_history = self.encode_history()
        agent_future = self.encode_future(history_enc)#context

        q_z_dist = self.get_posterior(agent_future)
        q_z_sample = q_z_dist.rsample()
        p_z_dist = self.get_prior(agent_history, sample_num=1)
        z = q_z_sample  # [N*1 32]
        recon_motion, _ = self.decode_future(z, 1, history_enc, dec_in=dec_in)
        p_z_dist_var = self.get_prior(agent_history, sample_num=sample_num)
        z_var = p_z_dist_var.sample()
        #
        var_motion, _ = self.decode_future(z_var, sample_num, history_enc, dec_in=dec_in)

        return self.fut_motion_orig, recon_motion, var_motion, q_z_dist, p_z_dist, self.fut_motion_mask
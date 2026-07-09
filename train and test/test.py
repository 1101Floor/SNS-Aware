import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.SNSAware import SNSAware
from model.sampler import Sampler
from utils.metrics import compute_ADE, compute_FDE
from model.model_baseline import TrajectoryModel

sys.path.append(os.getcwd())
from utils.torchutils import *
from utils.utils import prepare_seed, AverageMeter
import copy


parser = argparse.ArgumentParser()

# task setting
parser.add_argument('--obs_len', type=int, default=20)
parser.add_argument('--pred_len', type=int, default=12)

parser.add_argument('--sdd_scale', type=float, default=50.0)
parser.add_argument('--dataset', type=str, default='Lng', help='Lng')
parser.add_argument('--dataset_Mo', type=str, default='Mo', help='Mo')

parser.add_argument('--pos_concat', type=bool, default=True)
parser.add_argument('--cross_motion_only', type=bool, default=True)

parser.add_argument('--tf_model_dim', type=int, default=128)
parser.add_argument('--tf_ff_dim', type=int, default=128)
parser.add_argument('--tf_nhead', type=int, default=8)
parser.add_argument('--tf_dropout', type=float, default=0.1)

parser.add_argument('--he_tf_layer', type=int, default=2)
parser.add_argument('--fe_tf_layer', type=int, default=2)
parser.add_argument('--fd_tf_layer', type=int, default=2)

parser.add_argument('--he_out_mlp_dim', default=None)
parser.add_argument('--fe_out_mlp_dim', default=None)
parser.add_argument('--fd_out_mlp_dim', default=None)

parser.add_argument('--num_tcn_layers', type=int, default=1)
parser.add_argument('--asconv_layer_num', type=int, default=3)
parser.add_argument('--scene_emb_dim', type=int, default=16)

parser.add_argument('--pred_dim', type=int, default=2)

parser.add_argument('--pooling', type=str, default='mean')
parser.add_argument('--nz', type=int, default=32)
parser.add_argument('--sample_k', type=int, default=20)

parser.add_argument('--max_train_agent', type=int, default=100)
parser.add_argument('--rand_rot_scene', type=bool, default=True)
parser.add_argument('--discrete_rot', type=bool, default=False)

# basemodel
parser.add_argument('--n_tcn', type=int, default=2)
parser.add_argument('--d_emb', type=int, default=32)

# sampler architecture
parser.add_argument('--qnet_mlp', type=list, default=[512, 256])
parser.add_argument('--share_eps', type=bool, default=True)
parser.add_argument('--train_w_mean', type=bool, default=True)

# limit
parser.add_argument('--lon_min', type=float, default=1322232.3174)
parser.add_argument('--lon_max', type=float, default=1453961.4575)
parser.add_argument('--lat_min', type=float, default=7488211.9536)
parser.add_argument('--lat_max', type=float, default=7633914.7115)

# testing options
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--sample_num', type=int, default=50)

parser.add_argument('--sampler_epoch', type=int, default=10)
parser.add_argument('--vae_epoch', type=int, default=35)

args = parser.parse_args([])


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1]
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]

    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]

    return nodes_.squeeze()


def test(SNSAware_model, sampler, loader_test, traj_scale):
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()
    raw_data_dict = {}
    step = 0

    for cnt, batch in enumerate(loader_test):
        step += 1
        seq_name = 'data'
        frame_idx = int(batch.pop()[0])
        batch = [tensor[0].cuda() for tensor in batch]

        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        obs_loss_mask, pred_loss_mask, V_obs, A_obs, V_tr, A_tr, A_TPCA_obs, \
        A_TPCA_tr, A_vs_obs, A_vs_tr, v_image_obs, v_image_pred, reference_image, \
        reference_image_mask, obs_traj_1, V_obs_1 = batch

        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame_idx))
        sys.stdout.flush()

        V_obs_tmp = V_obs.unsqueeze(0).permute(0, 3, 1, 2)
        V_obs_abs_1 = obs_traj_1.unsqueeze(0).permute(0, 2, 3, 1)
        V_obs_tmp_1 = V_obs_1.unsqueeze(0).permute(0, 3, 1, 2)

        with torch.no_grad():
            SNSAware_model.set_data(
                obs_traj_rel, pred_traj_gt_rel, obs_loss_mask, pred_loss_mask,
                V_obs_tmp, A_obs.squeeze(), A_TPCA_obs.squeeze(), A_vs_obs.squeeze(),
                reference_image.squeeze(0), v_image_obs.squeeze(), V_obs_abs_1, V_obs_tmp_1
            )
            dec_motion, _, _, attn_weights = sampler.forward(SNSAware_model)

        dec_motion = dec_motion * traj_scale

        traj_gt = pred_traj_gt.transpose(1, 2) * traj_scale
        obs_traj_gt = obs_traj.transpose(1, 2) * traj_scale

        agent_traj = []
        agent_pre = []
        sample_motion = dec_motion.detach().cpu().numpy()

        V_x = seq_to_nodes(obs_traj.unsqueeze(0).data.cpu().numpy().copy())

        V_obs = seq_to_nodes(obs_traj_rel.unsqueeze(0).data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.copy(),
            V_x[0, :, :].copy()
        )

        V_y = seq_to_nodes(pred_traj_gt.unsqueeze(0).data.cpu().numpy().copy())
        V_tr = seq_to_nodes(pred_traj_gt_rel.unsqueeze(0).data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.copy(),
            V_x[-1, :, :].copy()
        )

        num_v, simp, pred_len, fea_len = (
            sample_motion.shape[0],
            sample_motion.shape[1],
            sample_motion.shape[2],
            sample_motion.shape[3]
        )

        sample_motion_transfer = torch.from_numpy(sample_motion).permute(1, 2, 0, 3)

        for i in range(sample_motion_transfer.shape[0]):
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                sample_motion_transfer[i, :, :, :].data.cpu().numpy().squeeze().copy(),
                V_x[-1, :, :].copy()
            )
            agent_pre.append(V_pred_rel_to_abs)

        for i in range(np.array(agent_pre).shape[2]):
            agent_traj.append(np.array(agent_pre)[:, :, i, :])
        agent_traj = np.array(agent_traj)

        traj_gt = traj_gt.detach().cpu().numpy()

        ade = compute_ADE(agent_traj, traj_gt)
        ade_meter.update(ade, n=SNSAware_model.agent_num)

        fde = compute_FDE(agent_traj, traj_gt)
        fde_meter.update(fde, n=SNSAware_model.agent_num)

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(obs_traj_gt.cpu())
        raw_data_dict[step]['trgt'] = copy.deepcopy(traj_gt)
        raw_data_dict[step]['pred'] = copy.deepcopy(agent_traj)

    return ade_meter, fde_meter, raw_data_dict


obs_seq_len = args.obs_len
pred_seq_len = args.pred_len

data_set_Mo = './dataset/' + args.dataset_Mo + '/'
data_set = './dataset/' + args.dataset + '/'

prepare_seed(args.seed)

torch.set_default_dtype(torch.float32)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)

traj_scale = 1.0

'''dset_test = TrajectoryDataset(
    data_dir=data_set+'test/',
    data_dir_Mo=data_set_Mo+'test/',
    obs_len=obs_seq_len,
    pred_len=pred_seq_len,
    skip=1,
    norm_lap_matr=True,
    left=args.lon_min,
    right=args.lon_max,
    bottom=args.lat_min,
    top=args.lat_max)
torch.save(dset_test, "./dataset/dset_test.pt")'''
dset_test = torch.load("./dataset/dset_test.pt", weights_only=False)

dset_test = DataLoader(
    dset_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

baseline_model = TrajectoryModel(
    number_asymmetric_conv_layer=args.n_tcn,
    embedding_dims=args.d_emb,
    number_gcn_layers=1,
    dropout=0,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    n_tcn=args.n_tcn,
    out_dims=2
).cuda()

SNSAware_model = SNSAware(args, baseline_model)
sampler = Sampler(args)

# load VAE model
vae_dir = './checkpoints/' + args.dataset + '/vae/'
all_vae_models = os.listdir(vae_dir)
if len(all_vae_models) == 0:
    print('VAE model not found!')

default_vae_model = 'model_%04d.p' % args.vae_epoch
if default_vae_model not in all_vae_models:
    default_vae_model = all_vae_models[-1]

cp_path = os.path.join(vae_dir, default_vae_model)
print('loading model from checkpoint: %s' % cp_path)
model_cp = torch.load(cp_path, map_location='cpu')
SNSAware_model.load_state_dict(model_cp)

SNSAware_model.set_device(device)
SNSAware_model.eval()

# load sampler model
sampler_dir = './checkpoints/' + args.dataset + '/sampler/'
all_sampler_models = os.listdir(sampler_dir)
if len(all_sampler_models) == 0:
    print('sampler model not found!')

default_sampler_model = 'model_%04d.p' % args.sampler_epoch
if default_sampler_model not in all_sampler_models:
    default_sampler_model = all_sampler_models[-1]

cp_path = os.path.join(sampler_dir, default_sampler_model)
model_cp = torch.load(cp_path, map_location='cpu')
sampler.load_state_dict(model_cp)
print('loading model from checkpoint: %s' % cp_path)

sampler.set_device(device)
sampler.eval()

_, _, raw_data_dict = test(SNSAware_model, sampler, dset_test, traj_scale)
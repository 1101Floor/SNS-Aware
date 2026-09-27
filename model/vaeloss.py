import torch

def compute_motion_mse(fut_motion_orig, train_dec_motion, fut_mask, weight):
    diff = fut_motion_orig - train_dec_motion
    mask = fut_mask  # all one
    diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum()
    loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def compute_z_kld(q_z_dist, p_z_dist, agent_num, min_clip, weight):
    loss_unweighted = q_z_dist.kl(p_z_dist).sum()   # sn=1
    loss_unweighted /= agent_num
    loss_unweighted = loss_unweighted.clamp_min_(min_clip)
    loss = loss_unweighted * weight
    return loss, loss_unweighted


def compute_sample_loss(fut_motion_orig, infer_dec_motion, fut_mask, weight):
    diff = infer_dec_motion - fut_motion_orig.unsqueeze(1)  # handle different shapes
    mask = fut_mask.unsqueeze(1).unsqueeze(-1)
    diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    loss_unweighted = loss_unweighted.mean()
    loss = loss_unweighted * weight
    return loss, loss_unweighted

def mercator_to_pixel(x, y, x_min, x_max, y_min, y_max, H, W):
    u = (x - x_min) * (W - 1) / (x_max - x_min)
    v = (y_max - y) * (H - 1) / (y_max - y_min)  # y翻转
    return u, v

def rel_to_abs_mercator(rel_dxy, last_obs_xy):
    if rel_dxy.dim() == 3:  # (B,T,2)
        return last_obs_xy[:, None, :] + torch.cumsum(rel_dxy, dim=1)

    elif rel_dxy.dim() == 4:  # (B,S,T,2)
        return last_obs_xy[:, None, None, :] + torch.cumsum(rel_dxy, dim=2)
    
def collision_penalty_mask_hard(pred_xy_merc, M, x_min, x_max, y_min, y_max, lambda_c=1.0):
    """
    pred_xy_merc: (B,T,2) 预测墨卡托坐标(x,y) meters
    M: (H,W) 二元mask，航道/可航行=1，不可航行=0
    """
    device = pred_xy_merc.device
    M = M.to(device).float()
    H, W = M.shape

    x = pred_xy_merc[..., 0]
    y = pred_xy_merc[..., 1]

    u, v = mercator_to_pixel(x, y, x_min, x_max, y_min, y_max, H, W)

    out = (u < 0) | (u > (W - 1)) | (v < 0) | (v > (H - 1))
    ui = u.clamp(0, W - 1).long()
    vi = v.clamp(0, H - 1).long()
    m_val = M[vi, ui]                 # (B,T) 取到0/1
    nonwalk = (m_val < 0.5)           # (B,T) True=不可航行

    C = (out | nonwalk).float()       # (B,T) 0/1
    Lc_unweighted = C.sum(dim=1).mean()
    Lc = lambda_c * Lc_unweighted
    return Lc, Lc_unweighted

def compute_vae_loss(args, fut_motion_orig, train_dec_motion, infer_dec_motion, fut_mask, q_z_dist, p_z_dist, obs_end, ref_mask,\
                    x_min, x_max, y_min, y_max):
    agent_num = len(fut_mask)  # need to check
    pre_abs = rel_to_abs_mercator(train_dec_motion, obs_end)
    Lc, Lc_unweighted = collision_penalty_mask_hard(pre_abs, ref_mask, x_min, x_max, y_min, y_max, lambda_c=1.0)
    mse_loss, mse_loss_uw = compute_motion_mse(fut_motion_orig, train_dec_motion, fut_mask, args.mse_weight)
    kld_loss, kld_loss_uw = compute_z_kld(q_z_dist, p_z_dist, agent_num, args.kld_min_clamp, args.kld_weight)
    var_loss, var_loss_un = compute_sample_loss(fut_motion_orig, infer_dec_motion, fut_mask, args.var_weight)
    total_loss = mse_loss + kld_loss + var_loss + Lc
    loss_dict = {'mse': mse_loss, 'kld': kld_loss, 'sample': var_loss, 'collision':Lc}
    loss_dict_uw = {'mse': mse_loss_uw, 'kld': kld_loss_uw, 'sample': var_loss_un, 'collision':Lc_unweighted}
    return total_loss, loss_dict, loss_dict_uw


import pickle
import glob
from torch.utils.data.dataloader import DataLoader
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import TrajectoryModel
from LowHighmodel import TrajectoryLowHighModel

import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'




def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    # Calculate ADEs and FDEs
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]

    # Calculate TCCs
    pred_best = pred[temp[:, -1, :].argmin(dim=0), :, range(pred.size(2)), :]
    pred_gt_stack = torch.stack([pred_best, gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return ADEs, FDEs, TCCs



def test(model, loader_test, KSTEPS=20):

    model.eval()
    ade_all, fde_all, tcc_all = [], [], []

    step =0
    pic_cnt = 0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
            V_obs.shape[2])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
            V_obs.shape[1])
        identity_st = torch.ones((1, V_obs.shape[1]*V_obs.shape[2], V_obs.shape[1]*V_obs.shape[2]), device='cuda') * \
                            torch.eye(V_obs.shape[1]*V_obs.shape[2], device='cuda')  # [1 obs_len*N obs_len*N]

        identity_st = identity_st.cuda()
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()

        identity = [identity_spatial, identity_temporal, identity_st]

        CVM = V_obs[:, -1:, :, 1:]

        V_pred = model(V_obs, obs_traj)  # A_obs <8, #, #>

        V_pred = V_pred + CVM

        # V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        #

        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

        ade_stack, fde_stack, tcc_stack = [], [], []

        # lhs_sample = torch.tensor(lhs(2, samples=20))
        # qr_seq = torch.stack([box_muller_transform(lhs_sample) for _ in range(mean.size(0))], dim=1).unsqueeze(
        #     dim=2).type_as(mean)
        # sample = mean + (torch.linalg.cholesky(cov) @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)

        # Evaluate trajectories
        V_absl = V_pred.cumsum(dim=1) + V_obs_traj[[-1], :, :]

        ADEs, FDEs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

        ade_stack.append(ADEs.detach().cpu().numpy())
        fde_stack.append(FDEs.detach().cpu().numpy())
        tcc_stack.append(TCCs.detach().cpu().numpy())

        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        tcc_all.append(np.array(tcc_stack))


    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde, mean_tcc = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_tcc




def evaluate_model(model_class, model_path, args_path, dataset_root, is_lowhigh=False):
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    # Data prep
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/' + args.dataset + '/'

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # independent of args batch size
        shuffle=False,
        num_workers=1)

    if is_lowhigh:
        model = model_class(embedding_dims=64, number_gcn_layers=1, dropout=0.1,
                            obs_len=8, pred_len=12, n_tcn=5).cuda()
    else:
        model = model_class(embedding_dims=64, number_gcn_layers=1, dropout=0.1,
                            obs_len=8, pred_len=12, n_tcn=5).cuda()

    model.load_state_dict(torch.load(model_path))

    print("Testing ....")
    ade_, fde_, raw_data_dict = test(model, loader_test)
    print("ade:", ade_, " fde:", fde_)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    return ade_, fde_




def main():
    KSTEPS = 20
    ade_ls = []
    fde_ls = []
    print('Number of samples:', KSTEPS)
    print("*" * 100)

    root_low = './checkpoints/MSDHGCN'
    root_lowhigh = './checkpoints/MSDHGCN-LowHigh'

    dataset = ['/eth', '/hotel', '/univ', '/zara1', '/zara2']

    paths_low = list(map(lambda x: root_low + x, dataset))
    paths_lowhigh = list(map(lambda x: root_lowhigh + x, dataset))

    for feta in range(len(paths_low)):
        path_low = paths_low[feta]
        path_lowhigh = paths_lowhigh[feta]

        # --- TrajectoryModel ---
        exps = glob.glob(path_low)
        print('Testing TrajectoryModel (only low-frequency):', exps)
        for exp_path in exps:
            print("Evaluating model:", exp_path)
            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            ade_, fde_ = evaluate_model(TrajectoryModel, model_path, args_path, path_low)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
        print("*" * 50)

        # --- TrajectoryLowHighModel ---
        exps = glob.glob(path_lowhigh)
        print('Testing TrajectoryLowHighModel (high- + low-frequency):', exps)
        for exp_path in exps:
            print("Evaluating model:", exp_path)
            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            ade_, fde_ = evaluate_model(TrajectoryLowHighModel, model_path, args_path, path_lowhigh, is_lowhigh=True)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
        print("*" * 100)


if __name__ == '__main__':
    main()
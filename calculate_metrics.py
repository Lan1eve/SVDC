import os
import numpy as np
import torch.nn as nn
import argparse
from gmflow.geometry import *
from gmflow.gmflow import GMFlow
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser()

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')


    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # inference on a directory
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')

    parser.add_argument(
        "--infer_w",
        default= '512',
        type=int
    )
    parser.add_argument(
        "--infer_h",
        default= '288',
        type=int
    )
    parser.add_argument("--local_rank", type=int, default=0)

    return parser

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))

class flow_warping_loss_align_test(nn.Module):
    def __init__(self, infer_h, infer_w, alpha=50):
        super(flow_warping_loss_align_test, self).__init__()
        self.alpha = alpha
        self.infer_h = infer_h
        self.infer_w = infer_w

    def forward(self, warp_rgb, rgb, warp_depth, depth, device):
        warp_depth = normalize_prediction_robust(warp_depth.squeeze(1),
                                                 torch.ones((warp_depth.shape[0], self.infer_h, self.infer_w)).to(
                                                     device)).unsqueeze(1)
        depth = normalize_prediction_robust(depth.squeeze(1),
                                            torch.ones((warp_depth.shape[0], self.infer_h, self.infer_w)).to(
                                                device)).unsqueeze(1)

        diff_depth = torch.abs(warp_depth - depth)
        diff_rgb = (warp_rgb - rgb) ** 2
        mask_rgb = torch.exp(-(self.alpha * diff_rgb))
        mask_rgb = torch.sum(mask_rgb, dim=1, keepdim=True)
        weight_diff = torch.mul(mask_rgb, diff_depth)
        loss_one_pair = 10 * torch.mean(weight_diff)

        return loss_one_pair

def pic_to_torch(img_path):
    # read img
    img = Image.open(img_path)
    img = img.resize((512,288))
    img_array = np.array(img,dtype = np.float32)
    torch_img = torch.from_numpy(img_array).permute(2,0,1)
    return torch_img

def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std
    return img0, img1

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    rmae = RMAE(pred, gt)
    return rmae,rmse,a1,a2,a3,sq_rel,silog


def RMAE(pred_dep, gt):
    rmae = np.mean(np.abs((pred_dep[gt > 0] - gt[gt > 0]) / gt[gt > 0]))
    return rmae

if __name__ == '__main__':
    device = torch.device("cuda:0")
    device_flow = torch.device('cuda:0')

    parser = get_args_parser()
    args = parser.parse_args()

    model_flow = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device_flow)
    model_flow = torch.nn.DataParallel(model_flow,device_ids=[0])
    model_flow = model_flow.module
    args.resume = "gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth"
    print('Load checkpoint: %s' % args.resume)
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(args.resume, map_location = 'cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_flow.load_state_dict(weights, strict=args.strict_resume)
    model_flow.to(device_flow)
    model_flow.eval()

    temloss = flow_warping_loss_align_test(int(args.infer_h),int(args.infer_w))

    length = 299
    datasets = ['DS','TT']
    for data in datasets:
        if data == 'DS':
            data_scale = 50
            times = 3
            sequence_list = ['0', '1', '2']
        if data == 'TT':
            data_scale = 30
            sequence_list = ['0', '1']
            times = 2


        rmae_tot = 0
        ewmae_tot = 0
        TEPE_tot = 0
        OPW_tot = 0
        rmse_tot = 0
        a1_tot = 0
        a2_tot = 0
        a3_tot = 0

        for sequence in sequence_list:
            rmae = 0
            rmse = 0
            OPW = 0
            TEPE = 0
            a1 = 0
            a2 = 0
            a3 = 0

            for i in range (length):
                k = i + 1
                gt_t_path = f'datasets/{data}_tuihua{sequence}/depth_gt/00{i:04d}_gt.npy'
                gt_tplus_path = f'datasets/{data}_tuihua{sequence}/depth_gt/00{k:04d}_gt.npy'

                pred_t_path = f"result/{data}_tuihua{sequence}/swin_single_win1/00{i:04d}_final.npy"
                pred_tplus_path = f"result/{data}_tuihua{sequence}/swin_single_win1/00{k:04d}_final.npy"

                color_t_path = f"datasets/{data}_tuihua{sequence}/color/00{i:04d}_rgb_left.jpg"
                color_tplus_path = f"datasets/{data}_tuihua{sequence}/color/00{k:04d}_rgb_left.jpg"

                pred_t = torch.from_numpy(np.load(pred_t_path)).squeeze().unsqueeze(0).unsqueeze(0).to(device_flow)
                pred_tplus = torch.from_numpy(np.load(pred_tplus_path)).squeeze().unsqueeze(0).unsqueeze(0).to(device_flow)
                rgb_t = pic_to_torch(color_t_path).unsqueeze(0).to(device_flow)
                rgb_tplus = pic_to_torch(color_tplus_path).unsqueeze(0).to(device_flow)

                rgb_t_norm, rgb_tplus_norm = normalize_img(rgb_t,rgb_tplus)

                results_dict = model_flow(rgb_tplus_norm, rgb_t_norm,
                                          attn_splits_list=args.attn_splits_list,
                                          corr_radius_list=args.corr_radius_list,
                                          prop_radius_list=args.prop_radius_list,
                                          pred_bidir_flow=args.pred_bidir_flow,
                                          )
                flow2to1 = results_dict['flow_preds'][-1]

                img1to2_seq = flow_warp(rgb_t, flow2to1, mask=False)
                outputs1to2 = flow_warp(pred_t, flow2to1, mask=False)

                tem_loss = temloss(img1to2_seq, rgb_tplus, outputs1to2, pred_tplus, device_flow)
                OPW += tem_loss.item()

                GT_t = np.load(gt_t_path)
                GT_t = Image.fromarray(GT_t.astype('float32'), mode='F')
                GT_t = GT_t.resize((512,288))
                GT_t = np.array(GT_t, dtype=np.float32)
                GT_t = np.clip(GT_t, 0.001, 10)
                pred_t = np.load(pred_t_path).squeeze()

                valid_mask = np.logical_and(GT_t > 0.001, GT_t < 8.1)
                test_RMAE,test_RMSE,test_a1,test_a2,test_a3,test_sqrel,test_silog = (compute_errors(GT_t[valid_mask], pred_t[valid_mask]))

                a1 += test_a1
                a2 += test_a2
                a3 += test_a3
                rmae += test_RMAE
                rmse += test_RMSE

                GT_t = np.load(gt_t_path)
                GT_t = Image.fromarray(GT_t.astype('float32'), mode='F')
                GT_tplus = np.load(gt_tplus_path)
                GT_tplus = Image.fromarray(GT_tplus.astype('float32'), mode='F')
                GT_t = GT_t.resize((512,288))
                GT_tplus = GT_tplus.resize((512, 288))
                GT_t = np.array(GT_t, dtype=np.float32)
                GT_tplus = np.array(GT_tplus, dtype=np.float32)

                pred_t = np.load(pred_t_path).squeeze()
                pred_tplus = np.load(pred_tplus_path).squeeze()
                valid_mask1 = np.logical_and(GT_t > 0, GT_t < 10.0)
                valid_mask2 = np.logical_and(GT_tplus > 0, GT_tplus < 10.0)
                valid_mask = np.logical_and(valid_mask1, valid_mask2)

                TEPE += np.sum((abs((GT_tplus[valid_mask] - GT_t[valid_mask]) - (
                            pred_tplus[valid_mask] - pred_t[valid_mask])))) / valid_mask.sum()

            rmae = rmae / length
            rmse = rmse / length
            TEPE = np.sum(TEPE) / length
            OPW = OPW / length

            a1 = a1/length
            a2 = a2/length
            a3 = a3/length

            rmse_tot += rmse
            rmae_tot += rmae
            TEPE_tot += TEPE
            a1_tot += a1
            a2_tot += a2
            a3_tot += a3
            OPW_tot += OPW
            print(f'mode {data}_{sequence} metrics are : < rmae :{rmae} TEPE :{TEPE} OPW :{OPW} >  rmse :{rmse} a1 is {a1},a2 is {a2},a3 is {a3}>'  )

        rmse_tot /= times
        rmae_tot /= times
        TEPE_tot /= times
        OPW_tot /= times

        a1_tot /= times
        a2_tot /= times
        a3_tot /= times

        print(
            f'mode {data}_tot metrics are : < rmse :{rmse_tot}  rmae :{rmae_tot}  TEPE :{TEPE_tot}  OPW :{OPW_tot} > , a1 is {a1_tot}, a2 is {a2_tot}, a3 is {a3_tot}')
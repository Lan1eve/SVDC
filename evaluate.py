import os
import numpy as np
import torch
import random
from tqdm import tqdm
from src.utils.model_io import load_weights
from src.dataloader.test_TT_DS_paper import TT_DS_compose_paper
from src.models.SVDC import SVDC_3frame

from src.config import args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def predict_tta(model, input_data, args):
    _, pred, _, _, _, _ = model(input_data)
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    return torch.Tensor(pred)


def eval(model, test_loader, args, device):
    if args.save_dir is not None and not os.path.exists(f'{args.save_dir}'):
        os.system(f'mkdir -p {args.save_dir}')
    with torch.no_grad():
        model.eval()
        pred_list = []
        pred_list_single_win1 = []
        for index, batch in enumerate(tqdm(test_loader)):
            gt = batch['depth'].to(device)
            img = batch['image'].to(device)
            input_data = {'rgb': img}
            additional_data = {}
            additional_data['hist_data'] = batch['additional']['hist_data'].to(device)
            additional_data['rect_data'] = batch['additional']['rect_data'].to(device)
            additional_data['mask'] = batch['additional']['mask'].to(device)
            additional_data['patch_info'] = batch['additional']['patch_info'].to(device)
            additional_data['sparse_depth'] = batch['additional']['sparse_depth'].to(device)
            additional_data['position'] = batch['additional']['position'].to(device)
            additional_data['grid_depth'] = batch['additional']['grid_depth'].to(device)
            input_data.update({
                'additional': additional_data
            })

            final = predict_tta(model, input_data, args) # t+1

            if index == 0:
                start = final[:,0] + final[:,1] + final[:,2]
                pred_list.append(start)
            elif index == 1:
                pred_list[-1] = pred_list[-1] + final[:,0]
                pred_list.append(final[:, 1])
                pred_list.append(final[:, 2])
            elif index == (len(test_loader) - 1):
                pred_list[-1] = pred_list[-1] + final[:,0] + final[:,1] + final[:,2]
            else:
                pred_list[-2] = pred_list[-2] + final[:,0]
                pred_list[-1] = pred_list[-1] + final[:,1]
                pred_list.append(final[:,2])

            if index % 3 == 1 :
                pred_list_single_win1.append(final[:, 0])
                pred_list_single_win1.append(final[:, 1])
                pred_list_single_win1.append(final[:, 2])

    os.makedirs(args.save_dir, exist_ok=True)
    swin_path = os.path.join(args.save_dir,'swin')
    swin_single_path_win1 = os.path.join(args.save_dir, 'swin_single_win1')


    print('swin_path is ',swin_path)
    os.makedirs(swin_path, exist_ok=True)
    os.makedirs(swin_single_path_win1, exist_ok=True)

    for i in range(len(pred_list)):
        if i == 0 or i == (len(pred_list) - 1):
            pred_list[i] = pred_list[i]/4.0
        elif i == 1 or i == (len(pred_list) - 2):
            pred_list[i] = pred_list[i]/2.0
        else:
            pred_list[i] = pred_list[i]/3.0

        np.save(os.path.join(swin_path,f"{i:06d}_final.npy"), pred_list[i])
    single_win_len = len(pred_list_single_win1)
    for i in range(single_win_len):
        np.save(os.path.join(swin_single_path_win1, f"{i:06d}_final.npy"), pred_list_single_win1[i])


if __name__ == '__main__':
    seed = 21
    set_seed(seed)

    device = torch.device('cuda:0')
    if args.dataset == 'paper_test_TT_DS':
        test_loader = TT_DS_compose_paper(args,'online_eval').data

    model = SVDC_3frame(n_bins=args.n_bins, min_val=args.min_depth,
                   max_val=args.max_depth, norm='linear').to(device)
    model = load_weights(model, args.weight_path)
    model = model.eval()

    eval(model, test_loader, args, device)
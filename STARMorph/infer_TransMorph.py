import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch

import numpy
from torch import serialization

# 新增：将numpy标量对象加入安全列表
serialization.add_safe_globals([numpy._core.multiarray.scalar])
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn as nn

def main():
    atlas_dir = '/root/autodl-tmp/TransMorph/IXI/TransMorph/IXI_data/atlas.pkl'
    test_dir = '/root/autodl-tmp/TransMorph/IXI/TransMorph/IXI_data/Val/'
    model_idx = -1
    weights = [1, 1]  # 和训练时保持一致
    model_folder = 'TransMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    
    if 'Val' in test_dir:
        csv_name = model_folder[:-1]+'_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], weights_only=False)['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    
    # ========== 修改1: 使用和训练时相同的插值方法 ==========
    reg_model = utils.register_model(config.img_size, 'nearest')  # 改为nearest用于分割
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')  # 保留bilinear用于可视化
    reg_model_bilin.cuda()
    # ====================================================
    
    # ========== 新增：初始化损失函数（和训练时一致） ==========
    criterion_ncc = losses.NCC_vxm()
    criterion_grad = losses.Grad3d(penalty='l2')
    criterions = [criterion_ncc, criterion_grad]
    # ======================================================
    
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    
    # ========== 新增：损失指标的记录器 ==========
    eval_loss_total = utils.AverageMeter()
    eval_loss_ncc = utils.AverageMeter()
    eval_loss_grad = utils.AverageMeter()
    # ==========================================
    
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)  # 保存完整output
            x_def, flow = output[0], output[1]  # 解包
            
            # ========== 新增：计算损失（和训练时一致） ==========
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            
            eval_loss_total.update(loss.item(), x.size(0))
            eval_loss_ncc.update(loss_vals[0].item(), x.size(0))
            eval_loss_grad.update(loss_vals[1].item(), x.size(0))
            # ==================================================
            
            # ========== 修改2: 使用和训练时完全相同的分割变形方式 ==========
            # 直接用nearest插值变形分割标签(和训练验证时一致)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # ==============================================================
            
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            
            # ========== 修改3: 使用和训练时相同的DSC计算函数 ==========
            # 优先尝试使用dice_val_VOI,如果没有则退回到dice_val
            try:
                dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            except AttributeError:
                # 如果没有dice_val_VOI函数,使用dice_val
                dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            # =========================================================
            
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            
            # ========== 修改：打印时加入损失信息 ==========
            print('Sample {}: Trans DSC: {:.4f}, Raw DSC: {:.4f}, Loss: {:.4f} (NCC: {:.6f}, Grad: {:.6f}), det<0: {:.6f}'.format(
                stdy_idx, dsc_trans.item(), dsc_raw.item(), loss.item(), 
                loss_vals[0].item(), loss_vals[1].item(),
                np.sum(jac_det <= 0)/np.prod(tar.shape)))
            # ============================================
            
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        # ========== 修改：最终打印加入损失统计 ==========
        print('\n' + '='*80)
        print('FINAL TEST RESULTS:')
        print('='*80)
        print('Deformed DSC: {:.3f} ± {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Raw DSC: {:.3f} ± {:.3f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        print('Total Loss: {:.4f} ± {:.4f}'.format(eval_loss_total.avg, eval_loss_total.std))
        print('  - NCC Loss: {:.6f} ± {:.6f}'.format(eval_loss_ncc.avg, eval_loss_ncc.std))
        print('  - Grad Loss: {:.6f} ± {:.6f}'.format(eval_loss_grad.avg, eval_loss_grad.std))
        print('Negative Jacobian: {:.6f} ± {:.6f}'.format(eval_det.avg, eval_det.std))
        print('='*80)
        # =============================================

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()

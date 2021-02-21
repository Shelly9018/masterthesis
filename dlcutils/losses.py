import torch
import torch.nn.functional as F
import scipy.stats
from dlcutils.softargmax2d import SoftArgmax2D


def skeleton_loss(pred, target, criterion=F.mse_loss, connectivity=None):
    len_gt = torch.norm(target[:, 0, :].float() - target[:, 1, :].float())
    len_pred = torch.norm(pred[:, 0, :].float() - pred[:, 1, :].float())
    # return F.l1_loss(len_gt, len_pred) # i think its the same as torch.norm(len_gt, len_pred)
    return criterion(len_gt, len_pred)

def compute_offset_loss(real_pos, y_pred, offset_pred, criterion=F.mse_loss):
    y_pred_in_image = y_pred * 8
    real_offset = real_pos - y_pred_in_image
    # return torch.abs(torch.sum(real_offset.flatten() - offset_pred))
    #return torch.nn.MSELoss()(real_offset.flatten(), offset_pred)
    return criterion(real_offset.flatten(), offset_pred)

def compute_repel_loss(real_pos, y_pred, offset_pred):
    #high if point gets too close to other landmark gt
    sig = 30

    y_pred_in_image = ((y_pred * 8) + 4).flatten() + offset_pred
    dist_real_1_to_pred_2 = torch.norm(real_pos.flatten()[:2] - y_pred_in_image[2:])
    repel_pred2_by_real1 = torch.tensor(scipy.stats.norm.pdf(float(dist_real_1_to_pred_2), 0, sig) / scipy.stats.norm.pdf(0, 0, sig))

    dist_real_2_to_pred_1 = torch.norm(real_pos.flatten()[2:] - y_pred_in_image[:2])
    repel_pred1_by_real2 = torch.tensor(scipy.stats.norm.pdf(float(dist_real_2_to_pred_1), 0, sig) / scipy.stats.norm.pdf(0, 0, sig))

    #if dist_real_2_to_pred_1 < 75 or dist_real_1_to_pred_2 < 75:
    #    print("MAX REPEL")

    total_loss = repel_pred2_by_real1 + repel_pred1_by_real2

    #print("REPEL:", total_loss)

    return total_loss


def intermediateSupervision_softArgmax_Heatmap_ceLoss_reductionMean_skeleton(pred, target, targetscoremap, metrics):
    # TODO Debug: temporarily disabled skeleton loss as we train wih t1 joint only to test new heatmaps
    softArgMax = SoftArgmax2D(softmax_temp=0.000001)
    pred_softargmax_tensor = softArgMax(pred)
    mse_loss_softargmax = F.mse_loss(pred_softargmax_tensor, target.float(), reduction='mean')
    ce_loss = F.binary_cross_entropy_with_logits(pred, targetscoremap)
    sk_loss = skeleton_loss(pred_softargmax_tensor, target)
    total_loss = ce_loss * 100 + mse_loss_softargmax * 0.01 + sk_loss * 0.01
    # print ("CE:", ce_loss * 100, "MSE:", mse_loss_softargmax * 0.01, "Skeleton:", sk_loss * 0.01)
    #    metrics['ce_loss'] += total_loss.data.cpu().numpy() * targetscoremap.size(0)
    return total_loss


def DLC_loss(gt_points_in_image, gt_heatmaps, pred_points_in_heatmap, pred_heatmaps, pred_offsets):

    # combines several of the here introduced losses into a total loss

    dlc_loss = intermediateSupervision_softArgmax_Heatmap_ceLoss_reductionMean_skeleton(
        pred_heatmaps,
        pred_points_in_heatmap,
        gt_heatmaps,
        {}
    )

    offset_loss = compute_offset_loss(
        gt_points_in_image,
        pred_points_in_heatmap,
        pred_offsets
    )

    repel_loss = compute_repel_loss(
        gt_points_in_image,
        pred_points_in_heatmap,
        pred_offsets
    )

    total_loss = dlc_loss + offset_loss * 0.0001 + repel_loss * 1000

    return total_loss


# tools for generating various types of images from tensors

import numpy as np
import cv2
import torch
from dlcutils.softargmax2d import SoftArgmax2D


def position_and_offset_to_absolute_coordinates(positions, offsets, stride):
    # converts coordinates in heatmap coordinates and the offset into coordinates in image space
    # if tensors are not flattened, they will get flattened to result in
    # x1 y1 x2 y2 .... xn yn order. So if you provide a non-flattened tensor, make sure
    # that a flattened version wil result in this coordinate order.
    # this is true if you provide a 1 x 1 x num_coordinates x 2 tensor where the last dimension holds x, y coordinates.
    # so for example 1 x 1 x 3 x 2 for three keypoints or anything else with batch and channel dimensions = 1.
    #
    # Output: The result will be reshaped to the same shape as the positions input.

    positions_flat = positions.flatten()
    offsets_flat = offsets.flatten()

    result_tensor = torch.zeros_like(positions_flat)

    # make sure both have the same number of entries. If you get an AssertionError here, then your
    # positions and offsets tensors did not have the same number of entries (keypoint coordinates).
    assert positions_flat.shape == offsets_flat.shape

    for keypoint in range(positions_flat.shape[0] // 2):

        # we assume 1st entry of each point is x, 2nd is y
        idx_x = keypoint * 2
        idx_y = idx_x + 1

        result_tensor[idx_x] = positions_flat[idx_x] * stride + offsets_flat[idx_x]
        result_tensor[idx_y] = positions_flat[idx_y] * stride + offsets_flat[idx_y]

    return result_tensor.reshape_as(positions)

def heatmap_tensor_to_image_list(heatmaps):
    # converts a num_heatmaps by x by y tensor to a list of heatmap images
    # converted to 0...255 range
    # currently scales all images to 0...255 from min to max.
    # this should be changed in the future for improved flexibility

    image_list = []

    squeezed_heatmaps = heatmaps.squeeze()

    for current_heatmap in range(squeezed_heatmaps.shape[0]):

        heatmap_torch = squeezed_heatmaps[current_heatmap, :]
        heatmap_np = heatmap_torch.squeeze().detach().cpu().numpy()
        heatmap_np = heatmap_np - np.min(heatmap_np)
        heatmap_np = ((heatmap_np / np.max(heatmap_np)) * 255).astype(np.uint8)

        image_list.append(heatmap_np)

    return image_list


def tensor2np_img(img_tensor: torch.tensor):
    """
    converts a 0...1 float tensor to a 0...255 uint8 numpy image, re-arranges dimensions from CHW to HWC
    """
    return (img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)



def save_overlay_image_and_heatmaps(img, gt_heatmap_list, pred_heatmap_list, gt_coordinates, pred_coordinates, filename):
    overlay_image = create_overlay_image_from_tensor(img, coords_pred=pred_coordinates, coords_gt=gt_coordinates)
    cv2.imwrite(filename, overlay_image)

    for idx, pred_heatmap in enumerate(pred_heatmap_list):
        new_filename = filename[:-4] + "_pred_heatmap_" + str(idx).zfill(2) + filename[-4:]
        cv2.imwrite(new_filename, pred_heatmap)

    for idx, gt_heatmap in enumerate(gt_heatmap_list):
        new_filename = filename[:-4] + "_gt_heatmap_" + str(idx).zfill(2) + filename[-4:]
        cv2.imwrite(new_filename, gt_heatmap)




def create_test_image(self, img, heatmap_pred, coords_pred, offset_pred, filepath):
    # call example:
    #        self.create_test_image(x, y_pred, pred_softargmax_tensor, offset_pred,
    #                               os.path.join(self.eval_root, "test_" + str(batch_idx).zfill(5) + ".png"))


    img = img[0, ...].unsqueeze(0)
    # generate images and score maps first so we can display and draw on them
    img_np = (img.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    res_img = img_np.copy()  # we need to copy the array as cv2 wont work directly on the image otherwise since it has a different memory layout
    marker_colors = [(0, 255, 0), (0, 0, 255)]
    for idx in range(self.num_joints):
        # might actually also be - offset_pred, depends on current offset computation -> lets check it maybe ;)
        res_img = cv2.drawMarker(res_img.copy(), (coords_pred[0, idx, 0] * self.stride + offset_pred[2 * idx],
                                                  coords_pred[0, idx, 1] * self.stride + offset_pred[2 * idx + 1]),
                                 color=marker_colors[idx],
                                 markerType=cv2.MARKER_DIAMOND, thickness=5, markerSize=20)
    print(coords_pred, offset_pred)
    cv2.imwrite(filepath, res_img)

    for batch_idx in range(heatmap_pred.shape[0]):  # well not needed yet but we might introduce bs > 1 soon
        for j in range(self.num_joints):
            # print(heatmap_pred.shape)
            current_joint_heatmap_pred = heatmap_pred[batch_idx, j, :, :]  # .unsqueeze(0)
            # print(current_joint_heatmap_pred.shape)
            heatmap_pred_np = (current_joint_heatmap_pred.squeeze().detach().cpu().numpy())
            # print(heatmap_pred_np.shape)
            heatmap_pred_np = heatmap_pred_np - np.min(heatmap_pred_np)
            heatmap_pred_np = ((heatmap_pred_np / np.max(heatmap_pred_np)) * 255).astype(np.uint8)
            cv2.imwrite(filepath[:-4] + "_heatmap_" + str(j).zfill(2) + ".png", heatmap_pred_np)


def create_debug_image(img, heatmap_gt, heatmap_pred, coords_gt, y_coords_img, offset_pred, filename):

    # call example
    # if False:
    #     self.create_debug_image(x, y, y_pred, y_coords, y_coords_img, offset_pred,
    #                             os.path.join(self.eval_root, "val_" + str(batch_idx).zfill(5) + ".png"))
    #     im = self.create_overlay_image_from_tensor(x, y_coords, y_pred)
    #     # self.log("val_img", [wandb.Image(im, caption="pred_01")])
    #     # self.logger.experiment.log({"examples": [wandb.Image(im, caption="Label")]})
    #     print("VALIDATION: pred:", pred_softargmax_tensor.flatten() * 8 + offset_pred, "true:", y_coords_img)
    #     print("offset:", offset_pred)
    #     # self.val_pts_list.append(y_coords_img, y_pred, offset_pred)

    softArgMax = SoftArgmax2D(softmax_temp=0.000001)
    coords_pred = softArgMax(heatmap_pred)

    # assuming a batch size of 1 so we can squeeze() it out.
    # if batch size is > 1 we need to iterate over the batch. (not implemented yet, we go for the 1st image in the batch only)
    # also we deal with 1 coordinate only atm, this needs to be extended as well (currently WIP, should be done soon

    # if batch_size > 1 we use the 1st entry of the batches only
    img = img[0, ...].unsqueeze(0)
    heatmap_gt = heatmap_gt[0, ...].unsqueeze(0)
    heatmap_pred = heatmap_pred[0, ...].unsqueeze(0)
    coords_gt = coords_gt[0, ...].unsqueeze(0)

    # generate images and score maps first so we can display and draw on them
    img_np = (img.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    heatmap_gt_np = (
        heatmap_gt.squeeze().detach().cpu().numpy())  # transpose with 3 channels if we get more than 1 keypoint and thus get more than 1 channel (batch size and channel get squeezed() out atm)
    heatmap_gt_np = heatmap_gt_np - np.min(heatmap_gt_np)
    heatmap_gt_np = ((heatmap_gt_np / np.max(heatmap_gt_np)) * 255).astype(np.uint8)
    heatmap_pred_np = (heatmap_pred.squeeze().detach().cpu().numpy())
    heatmap_pred_np = heatmap_pred_np - np.min(heatmap_pred_np)
    heatmap_pred_np = ((heatmap_pred_np / np.max(heatmap_pred_np)) * 255).astype(np.uint8)

    # gt_pos_x = int(coords_gt.squeeze()[0])
    # gt_pos_y = int(coords_gt.squeeze()[1])

    # draw GT and prediction markers on the image
    res_img = img_np.copy()  # we need to copy the array as cv2 wont work directly on the image otherwise since it has a different memory layout
    marker_colors = [(0, 255, 0), (0, 0, 255)]  # 2 colors for now, we need more later if we have > 2 joints
    # print(y_coords_img, y_coords_img.shape)
    # print(offset_pred, offset_pred.shape)
    for idx in range(self.num_joints):
        res_img = cv2.drawMarker(res_img.copy(),
                                 (y_coords_img.flatten()[2 * idx], y_coords_img.flatten()[2 * idx + 1]),
                                 color=marker_colors[idx],
                                 markerType=cv2.MARKER_CROSS, thickness=5, markerSize=20)
        res_img = cv2.drawMarker(res_img.copy(), (coords_pred[0, idx, 0] * self.stride + offset_pred[2 * idx],
                                                  coords_pred[0, idx, 1] * self.stride + offset_pred[2 * idx + 1]),
                                 color=marker_colors[idx],
                                 markerType=cv2.MARKER_DIAMOND, thickness=5, markerSize=20)

    # these 2 lines dont work anymore if we predict multiple landmarks
    # pred_pos_x = int(coords_pred.squeeze()[0])
    # pred_pos_y = int(coords_pred.squeeze()[1])

    # res_img = cv2.drawMarker(img_np.copy(), (gt_pos_x * 8,gt_pos_y * 8), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=20)
    # res_img = cv2.drawMarker(res_img.copy(), (pred_pos_x * 8, pred_pos_y * 8), color=(0, 255, 0),markerType=cv2.MARKER_DIAMOND, thickness=5, markerSize=20)

    # draw original heatmap
    res_gt = heatmap_gt_np[
        0, ...].copy()  # cv2.drawMarker(heatmap_gt_np.copy(), (gt_pos_x, gt_pos_x), color=(255, 255, 255), markerType=cv2.MARKER_DIAMOND, thickness=1, markerSize=10)

    # draw predicted heatmap (TODO: for now only 1 joint, we may extend it with auto-generated sublots for other joints later)
    res_pred = cv2.drawMarker(heatmap_pred_np[0, ...].copy(), (coords_pred[0, idx, 0], coords_pred[0, idx, 1]),
                              color=(128, 128, 128), markerType=cv2.MARKER_DIAMOND, thickness=1, markerSize=10)

    fig, (ax_img, ax_heatmap_gt, ax_heatmap_pred) = plt.subplots(1, 3, figsize=(18, 5))
    ax_img.imshow(res_img)
    ax_heatmap_gt.imshow(res_gt, cmap="gray")
    ax_heatmap_pred.imshow(res_pred, cmap="gray")

    plt.savefig(filename)
    plt.close("all")


def create_overlay_image_from_tensor(img_tensor, coords_pred=None, coords_gt=None):
    img_np = tensor2np_img(img_tensor)
    res_img = img_np.copy()  # we need to copy the array as cv2 wont work directly on the image otherwise since it has a different memory layout
    marker_colors = [(0, 255, 0), (0, 0, 255)]  # 2 colors for now, we need more later if we have > 2 joints

    # assuming coordinates are forwarded in a 1 x 1 x num_joints x 2 or anything else that will squeeze() down
    # to a num_joints x 2 tensor.
    # btw how can I assert tensor sizes?
    if coords_gt != None:

        coords_gt_squeezed = coords_gt.squeeze()

        for idx in range(coords_gt_squeezed.shape[0]):
            res_img = cv2.drawMarker(
                res_img.copy(),
                (coords_gt_squeezed[idx, 0], coords_gt_squeezed[idx, 1]),
                color=marker_colors[idx],
                markerType=cv2.MARKER_CROSS,
                thickness=5,
                markerSize=20
            )

    if coords_pred != None:

        coords_pred_squeezed = coords_pred.squeeze()

        for idx in range(coords_pred_squeezed.shape[0]):
            res_img = cv2.drawMarker(
                res_img.copy(),
                (coords_pred_squeezed[idx, 0], coords_pred_squeezed[idx, 1]),
                color=marker_colors[idx],
                markerType=cv2.MARKER_DIAMOND,
                thickness=5,
                markerSize=20
            )

    return res_img
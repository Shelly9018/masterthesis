from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import os
import math
import glob
from imgaug.augmentables import Keypoint, KeypointsOnImage
import skimage.draw
import matplotlib.pyplot as plt  # for debug plotting

from dlcutils.augmentation_pipeline import PipelineCreator

class GroundtruthCreator():
    def __init__(self, root, annotation_csv, idx, stride=8, scoremap_size=(64, 80)):
        self.root = root
        self.annotation_csv = annotation_csv
        self.idx = idx

        # self.eye_x = None
        # self.eye_y = None
        # self.eye_in_scoremap_x = None
        # self.eye_in_scoremap_y = None
        # self.eye = None

        self.coords_list = []  # stores coords in original image coordinates
        self.transformed_coords = None  # will be of type imgaug.KeypointsOnImage later
        self.coords_in_scoremap = []  # stores coords in score map (i.e. downsampled by stride)
        self.filename = None

        # self.nose_x = None
        # self.nose_y = None

        # self.leftear_x = None
        # self.leftear_y = None

        # self.rightear_x = None
        # self.rightear_y = None

        # self.tailbase_x = None
        # self.tailbase_y = None

        # self.nose = None
        # self.leftear = None
        # self.rightear = None
        # self.tailbase = None

        self.orgimg = None

        # Number of poses (here: Nose, Leftear, Rightear, Tailbase)
        self.num_joints = 2

        # (60, 80) is the output size of scoremap which is the output of the neural network
        # self.sizeOfScoremap = (64, 64)
        self.sizeOfScoremap = scoremap_size  # 64, 80 is 1/8 of 1280 x 1024 scaled by 0.5

        # The nominal stride by which the neural network downscales the input
        self.stride = stride
        self.half_stride = self.stride / 2

        self.scmap = None

    def loadImage(self):
        # filename_pos was 1 in firas original code for some reason and worked, but in our csv the path is in the 0th column
        # update: seems pandas or DLC adds an index as 1st column, that's why filename_pos is in index 1
        filename_pos = 1
        img_path = os.path.join(self.root, self.annotation_csv.iloc[self.idx, filename_pos])
        img_path = img_path.replace("\\", "/")
        # print (img_path)
        self.orgimg = cv2.imread(img_path)  # Get the input image
        self.filename = self.annotation_csv.iloc[self.idx, filename_pos].replace("\\", "/")
        img, h, w = self.transformImage(self.orgimg)

        # print("SHAPE:", img.shape)

        return img, h, w

    def transformImage(self, orgimg):
        h, w, _ = orgimg.shape

        # Expand the dimension of the orgimg because the augmentation pipeline
        # expects a 4D numpy array of shape (N, height, width, channels)
        img = np.expand_dims(orgimg, axis=0)
        return img, h, w

    def loadPoseCoord(self):
        # x and y locations of the poses which are saved in a csv file"
        # reworked to work with arbitrary number of joints
        # and not specific joint numbers and names
        # self.nose_x = self.annotation_csv.iloc[self.idx, 2]
        # self.nose_y = self.annotation_csv.iloc[self.idx, 3]
        # self.leftear_x = self.annotation_csv.iloc[self.idx, 4]
        # self.leftear_y = self.annotation_csv.iloc[self.idx, 5]
        # self.rightear_x = self.annotation_csv.iloc[self.idx, 6]
        # self.rightear_y = self.annotation_csv.iloc[self.idx, 7]
        # self.tailbase_x = self.annotation_csv.iloc[self.idx, 8]
        # self.tailbase_y = self.annotation_csv.iloc[self.idx, 9]

        num_keypoints = int((len(self.annotation_csv.columns) - 2) / 2)
        if num_keypoints % 2 > 2:
            raise ValueError(
                "Cannot compute number of keypoints from CSV, seems file format is different from current DLC-based format. Please check your CSV files.")

        for keypoint_idx in range(num_keypoints):
            keypoint_x = self.annotation_csv.iloc[self.idx, 2 + keypoint_idx * 2]
            keypoint_y = self.annotation_csv.iloc[self.idx, 2 + keypoint_idx * 2 + 1]
            self.coords_list.append([keypoint_x, keypoint_y])

        # self.eye_x = self.annotation_csv.iloc[self.idx, 2]
        # self.eye_y = self.annotation_csv.iloc[self.idx, 3]

    def convertPoseCoordToFloat(self):
        # self.nose_x = float(self.nose_x)
        # self.nose_y = float(self.nose_y)
        # self.leftear_x = float(self.leftear_x)
        # self.leftear_y = float(self.leftear_y)
        # self.rightear_x = float(self.rightear_x)
        # self.rightear_y = float(self.rightear_y)
        # self.tailbase_x = float(self.tailbase_x)
        # self.tailbase_y = float(self.tailbase_y)
        # self.eye_x = float(self.eye_x)
        # self.eye_y = float(self.eye_y)
        ...

    def getKeyPoints(self):
        """ Create the Keypoints for the Augmentation. Each Pose gets assigned to a specific Keypoint.
            These Keypoints are in addition to the input image (target image) also transformed.
            (Necessary when e.g. rotation or flipping the image)
        """
        # kps = KeypointsOnImage([
        #     #Keypoint(x=self.nose_x, y=self.nose_y),
        #     #Keypoint(x=self.leftear_x, y=self.leftear_y),
        #     #Keypoint(x=self.rightear_x, y=self.rightear_y),
        #     #Keypoint(x=self.tailbase_x, y=self.tailbase_y),
        #     Keypoint(x=self.eye_x, y=self.eye_y)
        # ],
        #     shape=self.orgimg.shape)  # Used orgimg.shape because img has an expanded dimension. Only want (height, width, channels).

        # MK Update: create keypoints with iterator
        kps = KeypointsOnImage(
            [
                Keypoint(x=a[0], y=a[1]) for a in self.coords_list
            ],
            shape=self.orgimg.shape
        )

        return kps

    def handleTestImages(self):
        """ The newimg is the old img and the pose coordinates stay the same """
        self.newimg = self.orgimg

        # self.nose = (self.nose_x, self.nose_y)
        # self.leftear = (self.leftear_x, self.leftear_y)
        # self.rightear = (self.rightear_x, self.rightear_y)
        # self.tailbase = (self.tailbase_x, self.tailbase_y)
        self.eye = (self.eye_x, self.eye_y)

    def applyAugmentation(self, img, pipeline):
        # Create the pipeline using the method augment_pipeline, and the probability for performing every individual augmentation methods is
        # given by 0.5. The Target image and keypoints are fed into the pipeline and extracted as newimg and newkps.

        kps = self.getKeyPoints()

        self.newimg, newkps = pipeline(images=img, keypoints=kps)

        # store augmented keypoints
        self.transformed_coords = newkps

        # Remove the previously added dimension (as we only have one image as input, not a batch.
        # The batch is created in the dataloader)
        self.newimg = np.squeeze(self.newimg, axis=0)

        # add distance encoding
        distance_encoding = False
        if distance_encoding:
            x_factor = np.pi / self.newimg.shape[0]
            for x in range(self.newimg.shape[0]):
                self.newimg[x, :, 0] = np.sin(x_factor * x) * 255

            y_factor = np.pi / self.newimg.shape[1]
            for y in range(self.newimg.shape[1]):
                self.newimg[:, y, 2] = np.sin(y_factor * y) * 255


        # Extract the new coordinates (transformed) coordinates of the image
        # self.nose = (newkps.keypoints[0].x, newkps.keypoints[0].y)
        # self.leftear = (newkps.keypoints[1].x, newkps.keypoints[1].y)
        # self.rightear = (newkps.keypoints[2].x, newkps.keypoints[2].y)
        # self.tailbase = (newkps.keypoints[3].x, newkps.keypoints[3].y)
        # self.eye = (newkps.keypoints[0].x, newkps.keypoints[0].y)

    def getScoreMap(self):
        return self.scmap

    def getAugmentedImage(self):
        return self.newimg

    def get_keypoint_coordinates_in_image(self):
        return self.transformed_coords.to_xy_array()

    def ScoreMapDrawer(self):
        # TODO: Hier muss jetzt auf multi landmark umgestellt werden
        # Create a groundtruth scoremap of the same size that comes out of the neural network
        self.scmap = np.zeros((self.sizeOfScoremap[0], self.sizeOfScoremap[1], self.num_joints))
        # joints = (self.nose, self.leftear, self.rightear, self.tailbase)
        # joints = (self.eye)

        use_gaussian_heatmap = True
        if use_gaussian_heatmap:

            dist_thresh, dist_thresh_sq, std, grid = self.getScoreMapParameters()

            # enumerate(joints) to enumerate([list(joints)]) as we have one entry only
            # for i, joint in enumerate(self.coords_list):
            for i, joint in enumerate(self.transformed_coords):
                if (math.isnan(joint.x) is not True and math.isnan(joint.y) is not True):
                    j_x = joint.x
                    j_y = joint.y

                    j_x_sm = round((j_x - self.half_stride) / self.stride)
                    j_y_sm = round((j_y - self.half_stride) / self.stride)

                    # store coords for softargmax (TODO: move this one to another function asap!)
                    # self.eye_in_scoremap_x = j_x_sm
                    # self.eye_in_scoremap_y = j_y_sm

                    self.coords_in_scoremap.append([j_x_sm, j_y_sm])

                    map_j = grid.copy()

                    # Distance between the joint point and each coordinate
                    dist = np.linalg.norm(grid - (j_y, j_x), axis=2) ** 2

                    # Apply a gaussian to each point in the image
                    scmap_j = np.exp(-dist / (2 * (std ** 2)))  # default mean=0, std = dist_thresh/4
                    scmap_j = scmap_j * np.where(dist > 1000, 0, 1)

                    # # generate negative image for other joint
                    # other_joint = self.transformed_coords[1-i]
                    #
                    # j_x = other_joint.x
                    # j_y = other_joint.y
                    #
                    # j_x_sm = round((j_x - self.half_stride) / self.stride)
                    # j_y_sm = round((j_y - self.half_stride) / self.stride)
                    #
                    #
                    # #self.coords_in_scoremap.append([j_x_sm, j_y_sm])
                    #
                    # map_j = grid.copy()
                    #
                    # # Distance between the joint point and each coordinate
                    # dist = np.linalg.norm(grid - (j_y, j_x), axis=2) ** 2
                    #
                    # # Apply a gaussian to each point in the image
                    # scmap_otherj = -np.exp(-dist / (2 * (std ** 2)))  # default mea
                    #
                    # final_map = scmap_j + scmap_otherj
                    # final_map = final_map - np.min(final_map)
                    # final_map = final_map / np.max(final_map)

                    self.scmap[..., i] = scmap_j
        else:
            # triangle-based distance encoding approch
            for i, joint in enumerate(self.transformed_coords):
                if (math.isnan(joint.x) is not True and math.isnan(joint.y) is not True):
                    j_x = joint.x
                    j_y = joint.y

                    j_x_sm = round((j_x - self.half_stride) / self.stride)
                    j_y_sm = round((j_y - self.half_stride) / self.stride)

                    self.coords_in_scoremap.append([j_x_sm, j_y_sm])

                    # this here works only if we have exactly 2 keypoints
                    other_joint = 1 - i
                    other_x = self.transformed_coords[other_joint].x
                    other_y = self.transformed_coords[other_joint].y

                    other_x_sm = round((other_x - self.half_stride) / self.stride)
                    other_y_sm = round((other_y - self.half_stride) / self.stride)

                    scmap_j = self.orientated_scoremap_drawer(j_x_sm, j_y_sm, other_x_sm, other_y_sm, 160, 128)



                    self.scmap[..., i] = scmap_j

    def orientated_scoremap_drawer(self, x0, y0, x1, y1, w, h):

        scoremap = np.zeros((h, w))

        a_x = x1 - x0
        a_y = y1 - y0

        # neue ecke 1 generieren. sf = scale_factor, sollte um .3 liegen
        # senkrechte in 2d -> x/y vertauschen, eines der Vorzeichen auf -1
        sf = .2

        b_x = a_y * sf
        b_y = -a_x * sf

        c_x = -a_y * sf
        c_y = a_x * sf

        # d ist die spitze des spiegeldreiecks

        d_x = -a_x * sf
        d_y = -a_y * sf

        center_x = x0
        center_y = y0

        rr, cc = skimage.draw.polygon(
            (a_y + center_y, b_y + center_y, d_y + center_y, c_y + center_y),
            (a_x + center_x, b_x + center_x, d_x + center_x, c_x + center_x)
        )

        distances = np.linalg.norm(np.array((rr - center_y, cc - center_x)), axis=0)
        distances = 1 - (distances / distances.max())
        distances = distances ** (2)

        scoremap[rr, cc] = distances  # np.linalg.norm(np.array((rr-center_y, cc-center_x)), axis=0)

        return scoremap

    def getScoreMapParameters(self):
        width = self.sizeOfScoremap[1]
        height = self.sizeOfScoremap[0]

        dist_thresh = float((width + height) / 6)
        dist_thresh_sq = dist_thresh ** 2

        std = dist_thresh  / 4

        # Create a meshgrid for each coordinate in the image, e.g. (1,2), (22,43) etc.
        grid = np.mgrid[:height, :width].transpose((1, 2, 0))

        # Upscale the grid coordinates by a factor of 8 (stride) and add half_stride to reduce the offset to the real coordinate
        # -> Done to have each coordinate in the scoremap map to a location in the original image size
        grid = grid * self.stride + self.half_stride

        return dist_thresh, dist_thresh_sq, std, grid

    def get_keypoint_coordinates_in_scoremap(self):
        # return self.eye_in_scoremap_x, self.eye_in_scoremap_y
        # TODO: soll das hier schon in einen Tensor gemappt werden?
        return self.coords_in_scoremap


class VideoFileDatasetWithoutGT(Dataset):
    # TODO: move augmentation and scmap generation to helper as its identical for all datasets
    def __init__(self, video_file_path):
        super().__init__()

        self.cap = cv2.VideoCapture(video_file_path)

    def __len__(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # for now, we ignore the idx and simply always return the next frame.
    # reason: indexing in a video takes long
    # TODO: add indexing so we can have random access, read at lower fps and whatever
    def __getitem__(self, idx):
        ret, img = self.cap.read()

        img = img.transpose(2, 0, 1).astype(np.float32)
        img = img / 255

        print(idx)
        return img


class ImageFolderDatasetWithoutGT(Dataset):
    # TODO: move augmentation and scmap generation to helper as its identical for all datasets
    def __init__(self, in_glob="/work/scratch/share/mk2mk/pig_frames/*.png", augmentation_ini_file=None):
        super().__init__()

        self.file_list = glob.glob(in_glob)
        self.file_list.sort()
        self.ini_file = None

    def __len__(self):
        return len(self.file_list)

    # for now, we ignore the idx and simply always return the next frame.
    # reason: indexing in a video takes long
    # TODO: add indexing so we can have random access, read at lower fps and whatever
    def __getitem__(self, idx):
        img = cv2.imread(self.file_list[idx])

        # TODO: this one should actually be read from the ini
        img = cv2.resize(img, (640, 512))

        img = img.transpose(2, 0, 1).astype(np.float32)
        img = img / 255

        # print(idx)
        return img, self.file_list[idx]


class OwnDataset(Dataset):
    def __init__(self, annotation_csv, root, augmentation_ini_file=None, stride=2, scoremap_size=(256, 320)):
        super().__init__()
        self.annotation_csv = pd.read_csv(annotation_csv)
        self.root = root
        self.augmentation_ini_file = augmentation_ini_file
        # self.train=True # aktuell ists eh für train/val/test gleich, da Augmentierung immer identisch.
        self.stride = stride  # heatmap stride, origimg gets scaled down by this factor to get heatmap dimensions
        self.scoremap_size = scoremap_size  # heatmap size

    def __len__(self):
        return len(self.annotation_csv)

    def augment_pipeline(self, height=None, width=None, augmentationprob=0.5):
        """ Implements Augmentation Pipeline. Different Augmentation Methods can be turned on in the augment.ini file """

        # config_file = "/work/scratch/kopaczka/git/ma_khader/DeepLabCutLightning/Configs/augment.ini"
        config_file = self.augmentation_ini_file

        pipelineCreator = PipelineCreator(config_file)
        pipelineCreator.addAugmentationToPipeline(augmentationprob)
        pipeline = pipelineCreator.getPipeline()

        return pipeline

    # TODO: Seems this one is weird as it transforms training data but not validation data even though
    # I've added transform code to the validation as well
    def get_scmap_gaussian(self, idx):
        """ Uses a gaussian heatmap. Highest probability is given to the pixel at pose location."""

        groundTruthCreator = GroundtruthCreator(self.root, self.annotation_csv, idx, stride=self.stride,
                                                scoremap_size=self.scoremap_size)
        img, h, w = groundTruthCreator.loadImage()
        groundTruthCreator.loadPoseCoord()
        # groundTruthCreator.convertPoseCoordToFloat()

        pipeline = self.augment_pipeline(height=h, width=w, augmentationprob=1.0)
        groundTruthCreator.applyAugmentation(img, pipeline)

        groundTruthCreator.ScoreMapDrawer()

        newimg = groundTruthCreator.getAugmentedImage()
        scmap = groundTruthCreator.getScoreMap()
        kp_coords = groundTruthCreator.get_keypoint_coordinates_in_scoremap()
        kp_coords_img = groundTruthCreator.get_keypoint_coordinates_in_image()
        fn = groundTruthCreator.filename

        # Return the augmented image and the respective ground truth scoremap
        return newimg, scmap, kp_coords, kp_coords_img, fn

    def transformImageAndMask(self, img, mask):
        img = self.transform(img)
        mask = self.transform(mask)

        # Normalize the image to the Pretrained imagenet data
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = normalize(img)

        return img, mask

    def __getitem__(self, idx):
        img, scmap, kp_coordinates, kp_in_image, filename = self.get_scmap_gaussian(idx)
        mask = scmap

        # print("Image loaded successfully")

        # if self.transform:
        #    img, mask = self.transformImageAndMask(img, mask)

        img = img.transpose(2, 0, 1).astype(np.float32)
        img = img / 255
        mask = mask.transpose(2, 0, 1).astype(np.float32)
        # print(idx)
        return [img, mask, np.array(kp_coordinates), kp_in_image, filename]

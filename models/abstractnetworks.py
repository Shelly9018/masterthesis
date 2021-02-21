import torch
import pytorch_lightning as pl
import csv


class AbstractKeypointNetwork(pl.LightningModule):
    """
    Abstract base class for all networks for general keypoint prediction.
    Implements basic functions common for all networks.
    """
    def __init__(self, network_name="AbstractKeypointNetwork"):
        super().__init__()

        self.network_name = network_name
        self.keypoint_names = ["KopfDB", "SchwanzDB"] # only set here for debug, should be overwritten

        self.val_keypoint_list_gt = []
        self.val_keypoint_list_pred = []
        self.test_keypoint_list_gt = []
        self.test_keypoint_list_pred = []

    def export_as_dlc_csv(self, keypoint_list, filename):
        # exports keypoints in accordance with DLC csv scheme
        csvfile = open(filename, 'w', newline='')
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)

        first_row = ["scorer"] + [self.network_name] * 2 * len(self.keypoint_names)
        second_row = ["bodyparts"] + [a for a in self.keypoint_names for _ in (0, 1)]
        third_row = ["coords"] + ["x", "y"] *  len(self.keypoint_names)

        csvwriter.writerow(first_row)
        csvwriter.writerow(second_row)
        csvwriter.writerow(third_row)


        # we assume that the networks have made sure then the rows contain [filename, x, y, x, y,...] so we
        # can simply dump em into the csv
        for row in keypoint_list:
            csvwriter.writerow(row)

        csvfile.close()


    def save_state_dict(self, p):
        torch.save(self.state_dict(), p)



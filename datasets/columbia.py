import json
import os
import random
from typing import List

import cv2
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

def gen_eye_mask(ldm, cam_id):
        """
        Calculate bounding box for eye region and save the image.
        :ldm: 2d face landamarks
        :image_gt: Image to analyse
        :save_path: Path used to save the generated eye mask
        """

        mask = np.zeros((512, 512))
        min_point = 0
        if ldm[24][1] < ldm[19][1]:
            min_point = 24
        else:
            min_point = 19
        max_point = 29
        if cam_id in [3, 4, 5, 6]:
            max_point = 33
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 51
        elif cam_id == 13:
            max_point = 62
        else:
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 30
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 33
            if (
                ldm[max_point][1] < ldm[40][1]
                or ldm[max_point][1] < ldm[41][1]
                or ldm[max_point][1] < ldm[46][1]
                or ldm[max_point][1] < ldm[47][1]
            ):
                max_point = 51
        mask[
            int(ldm[min_point][1]) : int(ldm[max_point][1]),
            int(ldm[17][0]) : int(ldm[26][0]),
        ] = 255

        return mask


def get_train_loader(
    data_dir,
    batch_size,
    num_workers=0,
    is_shuffle=True,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the train dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/columbia", "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "train"
    train_set = GazeDataset(
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_val_loader(
    data_dir,
    batch_size,
    num_val_images,
    num_workers=0,
    is_shuffle=True,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the validation dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/columbia", "train_test_split.json")
    print("load the val file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "val"
    val_set = GazeDataset(
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        num_val_images=num_val_images,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return val_loader


def get_test_loader(
    data_dir,
    batch_size,
    num_workers=0,
    is_shuffle=True,
    subject=None,
    evaluate=None,
    index_file=None,
):
    """
    Create the dataloader for the test dataset, takes the subjects from train_test_split.json .

    :data_dir: Path to the subjects files with all the information
    :batch_size: Batch size that will be used by the dataloader
    :num_workers: Num of workers to utilize
    :is_shuffle: Boolean value that indicates if images will be shuffled
    """

    # load dataset
    refer_list_file = os.path.join("data/columbia", "train_test_split.json")
    print("load the train file list from: ", refer_list_file)

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = "test"
    test_set = GazeDataset(
        dataset_path=data_dir,
        keys_to_use=datastore[sub_folder_use],
        sub_folder=sub_folder_use,
        transform=trans,
        is_shuffle=is_shuffle,
        subject=subject,
        evaluate=evaluate,
        index_file=index_file,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        keys_to_use: List[str] = None,
        sub_folder="",
        num_val_images=50,
        transform=None,
        is_shuffle=True,
        index_file=None,
        subject=None,
        evaluate=None,
    ):
        """
        Init function for the ETH-XGaze dataset, create key pairs to shuffle the dataset.

        :dataset_path: Path to the subjects files with all the information
        :keys_to_use: The subjects ID to use for the dataset
        :sub_folder: Indicate if it has to create the train,validation or test dataset
        :num_val_images: Used only for the validation dataset, indicate how many images to include in the validation dataset
        :transform: All the transformations to apply to the images
        :is_shuffle: Boolean value that indicates if images will be shuffled
        :index_file: Path to a specific key pairs file
        """

        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.evaluate = evaluate
        self.index_fix = None

        # Select keys
        if subject is None:
            self.selected_keys = [k for k in keys_to_use]
        else:
            self.selected_keys = [subject]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path,"columbia_" + self.selected_keys[num_i])
            print(self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, "r", swmr=True)
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                if sub_folder == "val":
                    n = num_val_images
                else:
                    n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [
                    (num_i, i) for i in range(n)
                ]  
        else:
            print("load the file: ", index_file[0])
            self.idx_to_kv = np.loadtxt(index_file[0], dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle and index_file is None:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.target_idx = np.loadtxt("configs/config_files/columbia_evaluation_target_single_subject.txt", dtype=np.int)
        

        self.hdf = None
        self.transform = transform

    def __len__(self):
        """
        Function that returns the length of the dataset.

        :return: Returns the length of the dataset
        """

        return len(self.idx_to_kv)

    def __del__(self):
        """
        Close all the hdfs files of the subjects.

        """

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def modify_index(self, index):
        self.index_fix = index

    def __getitem__(self, idx_input):
        """
        Return one sample from the dataset, the on in poistiongiven by idx.

        :idx: Indicate the position of the data sample to return
        :return: Returns one sample from the dataset
        """
        if self.index_fix is not None:
            idx_input = self.index_fix
        key, idx = self.idx_to_kv[idx_input]

        idx_input_image = idx
        key_input_image = key

        self.hdf = h5py.File(
            os.path.join(self.path,"columbia_" +  self.selected_keys[key]),
            "r",
            swmr=True,
        )
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf["face_patch"][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        face_mask = self.hdf["head_mask"][idx, :]
        left_eye_mask = self.hdf["left_eye_mask"][idx, :]
        right_eye_mask = self.hdf["right_eye_mask"][idx, :]

        kernel_2 = np.ones((3, 3), dtype=np.uint8)
        face_mask = cv2.erode(face_mask, kernel_2, iterations=2)

        ldms = self.hdf["facial_landmarks"][idx, :]
        cam_ind = self.hdf["cam_index"][idx, :]

        nl3dmm_para_dict = {}

        nl3dmm_para_dict["code"] = self.hdf["latent_codes"][0, :]
        nl3dmm_para_dict["code"][279:] = self.hdf["latent_codes"][idx, 279:]
        nl3dmm_para_dict["w2c_Rmat"] = self.hdf["w2c_Rmat"][idx, :]
        nl3dmm_para_dict["w2c_Tvec"] = self.hdf["w2c_Tvec"][idx, :]
        nl3dmm_para_dict["inmat"] = self.hdf["inmat"][idx, :]
        nl3dmm_para_dict["c2w_Rmat"] = self.hdf["c2w_Rmat"][idx, :]
        nl3dmm_para_dict["c2w_Tvec"] = self.hdf["c2w_Tvec"][idx, :]
        nl3dmm_para_dict["inv_inmat"] = self.hdf["inv_inmat"][idx, :]
        nl3dmm_para_dict["pitchyaw"] = self.hdf["pitchyaw_head"][idx, :]
        nl3dmm_para_dict["head_pose"] = self.hdf["face_head_pose"][idx, :]
        nl3dmm_para_dict["eye_mask"] = 0

        if self.evaluate == "target":
            idx = self.target_idx[idx_input]

            idx_target_image = idx
            key_target_image = key_input_image

            self.hdf = h5py.File(
                os.path.join(self.path,"columbia_" +  self.selected_keys[key]),
                "r",
                swmr=True,
            )
            assert self.hdf.swmr_mode

            # Get face image
            image_target = self.hdf["face_patch"][idx, :]
            image_target = image_target[:, :, [2, 1, 0]]  # from BGR to RGB
            image_target = self.transform(image_target)

            face_mask_target = self.hdf["head_mask"][idx, :]
            left_eye_mask_target = self.hdf["left_eye_mask"][idx, :]
            right_eye_mask_target = self.hdf["right_eye_mask"][idx, :]


            kernel_2_target = np.ones((3, 3), dtype=np.uint8)
            face_mask_target = cv2.erode(
                face_mask_target, kernel_2_target, iterations=2
            )

            ldms_target = self.hdf["facial_landmarks"][idx, :]
            cam_ind_target = self.hdf["cam_index"][idx, :]

            nl3dmm_para_dict_target = {}

            nl3dmm_para_dict_target["code"] = self.hdf["latent_codes"][0, :]
            nl3dmm_para_dict_target["code"][279:] = self.hdf["latent_codes"][idx, 279:]
            nl3dmm_para_dict_target["w2c_Rmat"] = self.hdf["w2c_Rmat"][idx, :]
            nl3dmm_para_dict_target["w2c_Tvec"] = self.hdf["w2c_Tvec"][idx, :]
            nl3dmm_para_dict_target["inmat"] = self.hdf["inmat"][idx, :]
            nl3dmm_para_dict_target["c2w_Rmat"] = self.hdf["c2w_Rmat"][idx, :]
            nl3dmm_para_dict_target["c2w_Tvec"] = self.hdf["c2w_Tvec"][idx, :]
            nl3dmm_para_dict_target["inv_inmat"] = self.hdf["inv_inmat"][idx, :]
            nl3dmm_para_dict_target["pitchyaw"] = self.hdf["pitchyaw_head"][idx, :]
            nl3dmm_para_dict_target["head_pose"] = self.hdf["face_head_pose"][idx, :]
            nl3dmm_para_dict_target["eye_mask"] = 0
            

            return (
                image,
                face_mask,
                left_eye_mask,
                right_eye_mask,
                nl3dmm_para_dict,
                ldms,
                cam_ind,
                idx_input_image,
                key_input_image,
                image_target,
                face_mask_target,
                left_eye_mask_target,
                right_eye_mask_target,
                nl3dmm_para_dict_target,
                ldms_target,
                cam_ind_target,
                idx_target_image,
                key_target_image,
            )
        elif self.evaluate == "landmark":
            return (
                image,
                face_mask,
                left_eye_mask,
                right_eye_mask,
                nl3dmm_para_dict,
                ldms,
                cam_ind,
            )
        else:
            return image, face_mask, left_eye_mask, right_eye_mask, nl3dmm_para_dict

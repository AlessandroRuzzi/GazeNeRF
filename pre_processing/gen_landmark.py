import argparse
from glob import glob

import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm

import os
from utils.logging import log_simple_image


class Gen2DLandmarks(object):
    def __init__(self, gpu_id, log) -> None:
        """
        Init function for Gen2DLandmarks.

        :gpu_id: Id of the GPU to use
        """

        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda:%s" % gpu_id
        else:
            self.device = "cpu"
        self.fa_func = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False, device=self.device
        )
        self.log = log

    def main_process(self, img_dir, previous_frame=None, columbia_path = None):
        """
        Function that generate 2d face landmarks, given an image direcotry.

        :img_dir: Path to the images to process, also used to save genereated landmarks
        """
        ldms_list = []
        if columbia_path is not None:
            img_path_list = [x for x in glob("%s/*.jpg" % img_dir) if "_mask" and "_big" not in x]
        else:
            img_path_list = [x for x in glob("%s/*.png" % img_dir) if "_mask" and "_big" not in x]

        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)

        img_path_list.sort()

        for img_path in tqdm(img_path_list, desc="Generate facial landmarks"):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if columbia_path is not None:
                img_rgb = img_rgb[:,864:4320]
                img_rgb = cv2.resize(img_rgb, (512, 512), interpolation = cv2.INTER_AREA)
            res = self.fa_func.get_landmarks(img_rgb)

            if res is None:
                print("Warning: can't predict the landmark info of %s" % img_path)
                if previous_frame is not None:
                    idx = img_path_list.index(img_path)
                    res = [previous_frame[idx]]
                else:
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.equalizeHist(img_gray)
                    res = self.fa_func.get_landmarks(img_gray)
                    
            if columbia_path is not None:
                image_name = os.path.basename(img_path)
                save_path = os.path.join(columbia_path,image_name[:-4] + "_lm2d.txt")
            else:
                save_path = img_path[:-4] + "_lm2d.txt"
            try:
                preds = res[0]
            except:
                ldms_list.append([])
                with open(save_path, "w") as f:
                    f.write("0")
                continue
            ldms_list.append(preds)
            if self.log and columbia_path is None:
                self.draw_landmarks(img_rgb, np.array(preds))
            with open(save_path, "w") as f:
                for tt in preds:
                    f.write("%f \n" % (tt[0]))
                    f.write("%f \n" % (tt[1]))
        return ldms_list

    def draw_landmarks(self, img, lms):
        """

        Function that draw landmakrs points on a given image, used to log images.

        :img: Image used to draw the landmarks
        :lms: 2d facial landmarks to draw
        """

        for i in range(lms.shape[0]):
            x = int(lms[i][0])
            y = int(lms[i][1])
            cv2.circle(img, (x, y), radius=2, color=(102, 102, 255), thickness=1)

        log_simple_image(img, "Facial Landmarks")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The code for generating facial landmarks."
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()

    tt = Gen2DLandmarks(args.gpu_id)
    tt.main_process(args.img_dir)

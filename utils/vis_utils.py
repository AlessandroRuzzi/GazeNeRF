import json
import os
import pickle as pkl

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import (QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap,
                         QTabletEvent)
from PyQt5.QtWidgets import (QApplication, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QListWidget, QMainWindow,
                             QPushButton, QSlider, QSplitter, QVBoxLayout,
                             QWidget)
from scipy.spatial.transform import Rotation as R

from configs.gazenerf_options import BaseOptions
from models.gaze_nerf import GazeNeRFNet
from utils.model_utils import Rmat2EulurAng, eulurangle2Rmat


class CustomQSlider(QWidget):
    def __init__(self, name, width, height, min_val, max_val, init_val=0) -> None:
        super().__init__()

        self.name = name
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.cur_val = init_val

        self.build_info()
        self.init_layout()

    def build_info(self):
        self.slider_w = int(self.width * 0.55)
        self.label_name_w = int(self.width * 0.16)
        self.label_val_w = int(self.width * 0.16)

    def update_geometry(self, width, height):
        self.slider_w = int(width * 0.55)
        self.label_name_w = int(width * 0.16)
        self.label_val_w = int(width * 0.16)
        self.slider.setFixedHeight(height)
        self.name_label.setFixedHeight(height)
        self.val_label.setFixedHeight(height)
        self.widget.setFixedHeight(height)

    def slider_set_value(self, cur_val):
        slider_val = int(
            self.slider_w * (cur_val - self.min_val) / (self.max_val - self.min_val)
        )
        self.slider.setValue(slider_val)
        self.cur_val = cur_val
        self.val_label.setText("%.02f" % self.cur_val)

    def init_layout(self):

        self.slider = QSlider()
        self.slider.setObjectName(self.name)

        self.slider.setFixedWidth(self.slider_w)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(0, self.slider_w)

        self.name_label = QLabel()
        self.name_label.setFixedWidth(self.label_name_w)
        self.name_label.setText(self.name)
        self.name_label.setAlignment(Qt.AlignLeft)
        self.name_label.setFont(QFont("Courier New", 13))
        self.name_label.setAlignment(Qt.AlignVCenter)

        self.val_label = QLabel()
        self.val_label.setFixedWidth(self.label_val_w)
        self.val_label.setFont(QFont("Courier New", 13))
        self.val_label.setAlignment(Qt.AlignVCenter)

        self.slider_set_value(self.cur_val)

        self.widget = QWidget()
        self.widget.setFixedHeight(self.height)

        self.h_layout = QHBoxLayout(self.widget)

        self.h_layout.addStretch(1)
        self.h_layout.addWidget(self.name_label)
        self.h_layout.addWidget(self.val_label)
        self.h_layout.addWidget(self.slider)
        self.h_layout.addStretch(1)
        self.h_layout.setAlignment(Qt.AlignVCenter)

    def update_labels(self):
        self.cur_val = (self.slider.value() / self.slider_w) * (
            self.max_val - self.min_val
        ) + self.min_val
        self.val_label.setText("%.02f" % self.cur_val)
        self.update()


class ArcBall:
    def __init__(self, NewWidth: float, NewHeight: float):
        self.StVec = np.zeros(3, "f4")  # Saved click vector
        self.EnVec = np.zeros(3, "f4")  # Saved drag vector
        self.AdjustWidth = 0.0  # Mouse bounds width
        self.AdjustHeight = 0.0  # Mouse bounds height
        self.setBounds(NewWidth, NewHeight)
        self.Epsilon = 1.0e-5

    def setBounds(self, NewWidth: float, NewHeight: float):
        assert (NewWidth > 1.0) and (NewHeight > 1.0)

        # Set adjustment factor for width/height
        self.AdjustWidth = 1.0 / ((NewWidth - 1.0) * 0.5)
        self.AdjustHeight = 1.0 / ((NewHeight - 1.0) * 0.5)

    def click(self, NewPt):  # Mouse down
        # Map the point to the sphere
        self._mapToSphere(NewPt, self.StVec)

    def drag(self, NewPt):  # Mouse drag, calculate rotation
        NewRot = np.zeros((4,), "f4")

        # Map the point to the sphere
        self._mapToSphere(NewPt, self.EnVec)

        # Return the quaternion equivalent to the rotation
        # Compute the vector perpendicular to the begin and end vectors
        Perp = np.cross(self.StVec, self.EnVec)

        # Compute the length of the perpendicular vector
        if np.linalg.norm(Perp) > self.Epsilon:  # if its non-zero
            # We're ok, so return the perpendicular vector as the transform
            # after all
            NewRot[:3] = Perp[:3]
            # In the quaternion values, w is cosine (theta / 2), where theta
            # is rotation angle
            NewRot[3] = np.dot(self.StVec, self.EnVec)
        else:  # if its zero
            # The begin and end vectors coincide, so return an identity
            # transform
            pass
        return NewRot

    def _mapToSphere(self, NewPt, NewVec):
        # Copy paramter into temp point
        TempPt = NewPt.copy()

        # Adjust point coords and scale down to range of [-1 ... 1]
        TempPt[0] = (TempPt[0] * self.AdjustWidth) - 1.0
        TempPt[1] = 1.0 - (TempPt[1] * self.AdjustHeight)

        # Compute the square of the length of the vector to the point from the
        # center
        length2 = np.dot(TempPt, TempPt)

        # If the point is mapped outside of the sphere...
        # (length^2 > radius squared)
        if length2 > 1.0:
            # Compute a normalizing factor (radius / sqrt(length))
            norm = 1.0 / np.sqrt(length2)

            # Return the "normalized" vector, a point on the sphere
            NewVec[0] = TempPt[0] * norm
            NewVec[1] = TempPt[1] * norm
            NewVec[2] = 0.0
        else:  # Else it's on the inside
            # Return a vector to a point mapped inside the sphere
            # sqrt(radius squared - length^2)
            NewVec[0] = TempPt[0]
            NewVec[1] = TempPt[1]
            NewVec[2] = np.sqrt(1.0 - length2)


class ArcBallUtil(ArcBall):
    def __init__(
        self, NewWidth: float, NewHeight: float, min_ang=-0.999, max_ang=0.999
    ):
        # self.Transform = np.identity(4, 'f4')
        self.LastRot = np.identity(3, "f4")
        self.ThisRot = np.identity(3, "f4")
        self.eulur_angle = np.zeros((3,), "f4")

        self.isDragging = False

        self.min_ang = min_ang
        self.max_ang = max_ang

        super().__init__(NewWidth, NewHeight)

    def resetRotation(self):
        self.isDragging = False
        self.LastRot = np.identity(3, "f4")
        self.ThisRot = np.identity(3, "f4")
        self.eulur_angle = np.zeros((3,), "f4")
        # self.Transform = self.Matrix4fSetRotationFromMatrix3f(self.Transform, self.ThisRot)

    def onClickLeftDown(self, cursor_x: float, cursor_y: float):
        # Set Last Static Rotation To Last Dynamic One
        self.LastRot = self.ThisRot.copy()
        # Prepare For Dragging
        self.isDragging = True
        mouse_pt = np.array([cursor_x, cursor_y], "f4")
        # Update Start Vector And Prepare For Dragging
        self.click(mouse_pt)
        return

    def onDrag(self, cursor_x, cursor_y):
        """Mouse cursor is moving"""
        if self.isDragging:
            mouse_pt = np.array([cursor_x, cursor_y], "f4")
            # Update End Vector And Get Rotation As Quaternion
            self.ThisQuat = self.drag(mouse_pt)
            # Convert Quaternion Into Matrix3fT
            self.ThisRot = self.Matrix3fSetRotationFromQuat4f(self.ThisQuat)

            temp_rot = np.matmul(self.LastRot, self.ThisRot)
            temp_eulur_angle = Rmat2EulurAng(temp_rot)

            if (
                np.sum(temp_eulur_angle > self.max_ang) > 0
                or np.sum([temp_eulur_angle < self.min_ang]) > 0
            ):
                #     # self.click(mouse_pt)
                return


            self.ThisRot = eulurangle2Rmat(temp_eulur_angle)


            self.eulur_angle = temp_eulur_angle
        return

    def onClickLeftUp(self):
        self.isDragging = False
        # Set Last Static Rotation To Last Dynamic One
        self.LastRot = self.ThisRot.copy()

    def Matrix3fSetRotationFromQuat4f(self, q1):
        if np.sum(np.dot(q1, q1)) < self.Epsilon:
            return np.identity(3, "f4")
        r = R.from_quat(q1)

        # transpose to make it identical to the C++ version
        return r.as_matrix()


class GazeNeRFUtils(object):
    def __init__(self, model_path) -> None:
        super().__init__()

        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = "cpu"

        self.source_img = None
        self.target_img = None

        self.build_net()
        self.build_cam()
        self.gen_uv_xy_info()

    def build_net(self):

        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        self.opt = BaseOptions(None)

        net = GazeNeRFNet(self.opt, include_vd=False, hier_sampling=False)
        net.load_state_dict(check_dict["net"])

        self.net = net.to(self.device)
        self.net.eval()

    def build_cam(self):

        with open("configs/config_files/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        # temp_inmat = torch.as_tensor(temp_dict["inmat"])
        temp_inv_inmat = torch.as_tensor(temp_dict["inv_inmat"])
        scale = self.opt.featmap_size / 32.0
        temp_inv_inmat[:2, :2] /= scale

        self.inv_inmat = temp_inv_inmat.view(1, 3, 3).to(self.device)

        tv_z = 0.5 + 11.5
        base_rmat = torch.eye(3).float().view(1, 3, 3).to(self.device)
        base_rmat[0, 1:, :] *= -1
        base_tvec = torch.zeros(3).float().view(1, 3, 1).float().to(self.device)
        base_tvec[0, 2, 0] = tv_z

        self.base_c2w_Rmats = base_rmat
        self.base_c2w_Tvecs = base_tvec

    def gen_uv_xy_info(self):
        mini_h = self.opt.featmap_size
        mini_w = self.opt.featmap_size

        indexs = torch.arange(mini_h * mini_w)
        x_coor = (indexs % mini_w).view(-1)
        y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)

        xy = torch.stack([x_coor, y_coor], dim=0).float()
        uv = torch.stack(
            [x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1
        )

        self.xy = xy.unsqueeze(0).to(self.device)
        self.uv = uv.unsqueeze(0).to(self.device)

    def exact_code(self, code_pkl_path):
        assert os.path.exists(code_pkl_path)
        temp_dict = torch.load(code_pkl_path, map_location="cpu")
        code_info = temp_dict["code"]

        for k, v in code_info.items():
            if v is not None:
                code_info[k] = v.to(self.device)

        shape_code = code_info["shape_code"]
        iden_code = shape_code[:, :100]
        expr_code = shape_code[:, 100:]

        appea_code = code_info["appea_code"]
        text_code = appea_code[:, :100]
        illu_code = appea_code[:, 100:]

        return iden_code, expr_code, text_code, illu_code

    def build_code(self, config_path):
        assert os.path.exists(config_path)
        with open(config_path) as f:
            temp_dict = json.load(f)

        iden_code_1, expr_code_1, text_code_1, illu_code_1 = self.exact_code(
            temp_dict["code_path_1"]
        )
        self.iden_code_1 = iden_code_1
        self.expr_code_1 = expr_code_1
        self.text_code_1 = text_code_1
        self.illu_code_1 = illu_code_1

        iden_code_2, expr_code_2, text_code_2, illu_code_2 = self.exact_code(
            temp_dict["code_path_2"]
        )
        self.iden_code_2 = iden_code_2
        self.expr_code_2 = expr_code_2
        self.text_code_2 = text_code_2
        self.illu_code_2 = illu_code_2

    def update_code_1(self, file_path):
        assert os.path.exists(file_path)
        iden_code_1, expr_code_1, text_code_1, illu_code_1 = self.exact_code(file_path)
        self.iden_code_1 = iden_code_1
        self.expr_code_1 = expr_code_1
        self.text_code_1 = text_code_1
        self.illu_code_1 = illu_code_1

        gaze_code = torch.FloatTensor([[500, 400]]) / 1024
        shape_code = torch.cat([self.iden_code_1, self.expr_code_1, gaze_code], dim=1)
        appea_code = torch.cat([self.text_code_1, self.illu_code_1], dim=1)

        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
        }
        cam_info = self.gen_cam(0.0, 0.0, 0.0)

        pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        self.source_img = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )

    def update_code_2(self, file_path):
        assert os.path.exists(file_path)
        iden_code_2, expr_code_2, text_code_2, illu_code_2 = self.exact_code(file_path)
        self.iden_code_2 = iden_code_2
        self.expr_code_2 = expr_code_2
        self.text_code_2 = text_code_2
        self.illu_code_2 = illu_code_2

        gaze_code = torch.FloatTensor([[500, 400]]) / 1024
        shape_code = torch.cat([self.iden_code_2, self.expr_code_2, gaze_code], dim=1)
        appea_code = torch.cat([self.text_code_2, self.illu_code_2], dim=1)

        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
        }
        cam_info = self.gen_cam(0.0, 0.0, 0.0)

        pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        self.target_img = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )

    def gen_code(self, iden_t, expr_t, text_t, illu_t):
        iden_code = self.iden_code_1 * (1 - iden_t) + self.iden_code_2 * iden_t
        text_code = self.text_code_1 * (1 - text_t) + self.text_code_2 * text_t
        illu_code = self.illu_code_1 * (1 - illu_t) + self.illu_code_2 * illu_t
        expr_code = self.expr_code_1 * (1 - expr_t) + self.expr_code_2 * expr_t

        gaze_code = torch.FloatTensor([[500, 400]]) / 1024
        shape_code = torch.cat([iden_code, expr_code, gaze_code], dim=1)
        appea_code = torch.cat([text_code, illu_code], dim=1)

        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
        }

        return code_info

    def gen_cam(self, pitch, yaw, roll):

        angle = np.array([-pitch, -yaw, -roll])
        delta_rmat = eulurangle2Rmat(angle)
        delta_rmat = torch.from_numpy(delta_rmat).unsqueeze(0).to(self.device)

        new_rmat = torch.bmm(delta_rmat, self.base_c2w_Rmats)
        new_tvec = torch.bmm(delta_rmat, self.base_c2w_Tvecs)
        cam_info = {
            "batch_Rmats": new_rmat,
            "batch_Tvecs": new_tvec,
            "batch_inv_inmats": self.inv_inmat,
        }

        return cam_info

    def gen_image(self, iden_t, expr_t, text_t, illu_t, pitch, yaw, roll):
        code_info = self.gen_code(iden_t, expr_t, text_t, illu_t)
        cam_info = self.gen_cam(pitch, yaw, roll)


        pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)
        img = pred_dict["coarse_dict"]["merge_img"]
        img = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img


class AxisUtils(object):
    def __init__(self, img_size) -> None:

        self.set_img_size(img_size)
        self.build_info()
        self.build_cam()

    def set_img_size(self, img_size):
        self.img_size = img_size

        with open("configs/config_files/cam_inmat_info_32x32.json", "r") as f:
            temp_dict = json.load(f)
        inmat = temp_dict["inmat"]

        scale = self.img_size / 32.0
        self.inmat = np.array(inmat)
        self.inmat[:2, :] *= scale

        self.fx = self.inmat[0, 0]
        self.fy = self.inmat[1, 1]
        self.cx = self.inmat[0, 2]
        self.cy = self.inmat[1, 2]

    def build_info(self):

        length = 0.75
        self.origin = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_x = np.array([length, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_y = np.array([0.0, length, 0.0], dtype=np.float32).reshape(3, 1)
        self.axis_z = np.array([0.0, 0.0, length], dtype=np.float32).reshape(3, 1)

    def build_cam(self):

        base_rmat = np.eye(3).astype(np.float32)
        base_rmat[1:, :] *= -1

        tv_z = 0.5 + 11.5
        base_tvec = np.zeros((3, 1), dtype=np.float32)
        base_tvec[2, 0] = tv_z

        self.base_w2c_Rmats = base_rmat
        self.base_w2c_Tvecs = base_tvec

        self.cur_Rmat = base_rmat.copy()
        self.cur_Tvec = base_tvec.copy()

    def calc_proj_pts(self, vp):
        cam_vp = self.cur_Rmat.dot(vp) + self.cur_Tvec
        u = self.fx * (cam_vp[0, 0] / cam_vp[2, 0]) + self.cx
        v = self.fy * (cam_vp[1, 0] / cam_vp[2, 0]) + self.cy

        return (int(u), int(v))

    def update_CurCam(self, pitch, yaw, roll):
        angles = np.zeros(3, dtype=np.float32)

        angles[0] = -pitch
        angles[1] = -yaw
        angles[2] = -roll

        delta_rmat = eulurangle2Rmat(angles)

        # c2w
        self.cur_Rmat = delta_rmat.dot(self.base_w2c_Rmats)
        self.cur_Tvec = delta_rmat.dot(self.base_w2c_Tvecs)

        # w2c
        self.cur_Tvec = -(self.cur_Rmat.T.dot(self.cur_Tvec))
        self.cur_Rmat = self.cur_Rmat.T

    def generate_img(self, pitch, yaw, roll, img=None):
        self.update_CurCam(pitch, yaw, roll)

        if img is None:
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 255

        pixel_o = self.calc_proj_pts(self.origin)
        pixel_x = self.calc_proj_pts(self.axis_x)
        pixel_y = self.calc_proj_pts(self.axis_y)
        pixel_z = self.calc_proj_pts(self.axis_z)

        img = img.copy()

        img = cv2.arrowedLine(img, pixel_o, pixel_x, color=(255, 0, 0), thickness=1)
        img = cv2.arrowedLine(img, pixel_o, pixel_y, color=(0, 255, 0), thickness=1)
        img = cv2.arrowedLine(img, pixel_o, pixel_z, color=(0, 0, 255), thickness=1)

        return img

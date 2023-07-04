######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
######################################################################################################################################

Here you can find the contents for the MPIIFaceGaze dataset.

- Data
There are 15 participants and corresponding folder from p00 to p14. Each folder includes the images inside the different day folder.

- Annotation
There are pxx.txt file in each participant folder. which saves the information:
Dimension 1: image file path and name.
Dimension 2~3: Gaze location on the screen coordinate in pixels, the actual screen size can be found in the "Calibration" folder.
Dimension 4~15: (x,y) position for the six facial landmarks, which are four eye corners and two mouth corners.
Dimension 16~21: The estimated 3D head pose in the camera coordinate system based on 6 points-based 3D face model, rotation and translation: we implement the same 6 points-based 3D face model in [1], which includes the four eye corners and two mouth corners
Dimension 22~24 (fc): Face center in the camera coordinate system, which is averaged 3D location of the 6 focal landmarks face model. Not it is slightly different with the head translation due to the different centers of head and face.
Dimension 25~27 (gt): The 3D gaze target location in the camera coordinate system. The gaze direction can be calculated as gt - fc. 
Dimension 28: Which eye (left or right) is used for the evaluation subset in [2].

- Calibration
There is the "Calibration" folder for each participant, which contains
(1)Camera.mat: the intrinsic parameter of the laptop camera. "cameraMatrix": the projection matrix of the camera. "distCoeffs": camera distortion coefficients. "retval": root mean square (RMS) re-projection error. "rvecs": the rotation vectors. "tvecs": the translation vectors.
(2) monitorPose.mat: the position of image plane in camera coordinate. "rvecs": the rotation vectors. "tvecs": the translation vectors.
(3) creenSize.mat: the laptop screen size. "height_pixel": the screen height in pixel. "width_pixel": the screen width in pixel. "height_mm": the screen height in millimeter. "width_mm": the screen width in millimeter.


Enjoy!

[1] Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-synthesis for appearance-based 3d gaze estimation. In Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on, pages 1821–1828. IEEE, 2014.
[2] X. Zhang, Y. Sugano, M. Fritz and A. Bulling. Appearance-based Gaze Estimation in the Wild. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
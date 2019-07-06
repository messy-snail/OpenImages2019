import pandas as pd
import numpy as np
from random import *
import collections

from matplotlib import pyplot as plt
from PIL import Image
import glob

import cv2
import sys
import os
import numpy as np
import random

import skimage.io
import skimage.transform
import skimage.color
import skimage
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from operator import itemgetter

"""
https://pytorch.org/docs/0.2.0/torchvision/transforms.html
https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
"""


class IMGRTDataset(Dataset):
    """IMG and RT dataset loader"""

    def __init__(self, train_file, transform=None):

        self.DB_path = train_file[0]
        self.folder_name = train_file[1]

        self.transform = transform

        self.debug_name_check = []

        # ORB Detector
        self.orb = cv2.ORB_create(500)
        # self.MIN_MATCH_COUNT = 10
        self.MIN_MATCH_COUNT = 5

        # Debug
        # tmp = cv2.imread("loss.png", cv2.IMREAD_GRAYSCALE)
        # fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # axs[0].set_title('img1')
        # axs[0].imshow(img1)
        #
        # axs[1].set_title('img2')
        # axs[1].imshow(img2)
        # plt.show()

        try:
            self.image_path_list, self.image_cnt_len = self._read_annotations()

        except ValueError as e:
            raise_from(ValueError('_read_annotations error: {}: {}'.format(self.DB_path, e)), None)

        # keys() : 딕셔너리의 키를 모아서 dict_keys 객체 리턴
        # image_data의 키를 리스트화해서 image_names에 저장, 영상 name list라고 보면 됨

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        return result

    def __getitem__(self, idx):
        # stuff
        """
        __getitem__() function returns the data and labels.
        This function is called from dataloader like this:
        img, label = blah_blah_blah_Dataset.__getitem__(99)  # For 99th item
        So, the idx parameter is the nth data/image (as tensor) you are going to return.

        An important thing to note is that __getitem__() return a specific type for a single data point (like a tensor, numpy array etc.), otherwise, in the data loader you will get an error like:

        TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>
        """

        self.name, self.frame = self.find_param(self.image_path_list[idx])

        # St-1
        St_1_33, _, cv_old_frame = self.load_image(idx, flag=0, frame=self.frame-33)
        st_1_33, _, _ = self.load_image(idx, flag=0, frame=self.frame - 33)
        st_1_17, _, _ = self.load_image(idx, flag=0, frame=self.frame - 17)
        st_1_9, _, _ = self.load_image(idx, flag=0, frame=self.frame - 9)
        st_1_5, _, _ = self.load_image(idx, flag=0, frame=self.frame - 5)
        st_1_3, _, _ = self.load_image(idx, flag=0, frame=self.frame - 3)
        st_1_2, _, _ = self.load_image(idx, flag=0, frame=self.frame - 2)

        # It-1
        it_1_1, _, _ = self.load_image(idx, flag=1, frame=self.frame - 1)

        # St-0
        st_0_32, _, _ = self.load_image(idx, flag=0, frame=self.frame - 32)
        st_0_16, _, _ = self.load_image(idx, flag=0, frame=self.frame - 16)
        st_0_8, _, _ = self.load_image(idx, flag=0, frame=self.frame - 8)
        st_0_4, _, _ = self.load_image(idx, flag=0, frame=self.frame - 4)
        st_0_2, _, _ = self.load_image(idx, flag=0, frame=self.frame - 2)
        st_0_1, _, cv_old_sframe = self.load_image(idx, flag=0, frame=self.frame - 1)

        # It-0
        it_0_0, _, cv_old_frame = self.load_image(idx, flag=1, frame=self.frame - 0)

        # Ground Truth frame (St_0_0)
        st_0_0, _, cv_frame = self.load_image(idx, flag=0, frame=self.frame - 0)

        set_t_1 = {'s_33': st_1_33, 's_17': st_1_17, 's_9': st_1_9, 's_5': st_1_5, 's_3': st_1_3, 's_2': st_1_2,
                   'i_1': it_1_1}

        set_t_0 = {'s_32': st_0_32, 's_16': st_0_16, 's_8': st_0_8, 's_4': st_0_4, 's_2': st_0_2, 's_1': st_0_1,
                   'i_0': it_0_0}

        kp0, des0 = self.orb.detectAndCompute(cv_old_sframe, None)
        kp1, des1 = self.orb.detectAndCompute(cv_old_frame, None)
        kp2, des2 = self.orb.detectAndCompute(cv_frame, None)


        # Brute Force Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        '''
        crossCheck is an alternative to the ratio test. It breaks knnMatch. 
        So either use crossCheck=False and then do the ratio test, or use crossCheck=True and use bf.match() instead of bf.knnMatch().
        '''

        # matches = bf.match(self.des1, self.des2)
        matches = bf.knnMatch(des1, des2, k=2)

        tp_matches = bf.knnMatch(des0, des2, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)

        # Apply Lowe Ratio Test to the keypoints
        # this should weed out unsure matches
        good_kpt = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_kpt.append(m)

        tp_good_kpt = []
        for m, n in tp_matches:
            if m.distance < 0.7 * n.distance:
                tp_good_kpt.append(m)

        feature_dist = 0
        diff_RT_Mat = 0
        M = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        tp_M = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        # flag = False
        new_src_pts = np.zeros((500, 1, 3))
        new_dst_pts = np.zeros((500, 1, 2))
        pt_num = 0

        if len(good_kpt) > self.MIN_MATCH_COUNT:
            # print("matches are found - %d/%d" % (len(good_kpt), self.MIN_MATCH_COUNT))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_kpt]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_kpt]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = cv_old_frame.shape

            pt_num, _, _ = src_pts.shape

            matchesMask = mask.ravel().tolist()


            tmp_src_pts = src_pts.squeeze()
            shallow_copy_src = tmp_src_pts.transpose()
            shallow_copy_src[0] /= w
            shallow_copy_src[1] /= h
            ones = np.ones((len(src_pts), 1))
            new_src_pts = np.hstack([tmp_src_pts, ones])


            new_dst_pts = dst_pts.squeeze()
            shallow_copy_dst = new_dst_pts.transpose()
            shallow_copy_dst[0] /= w
            shallow_copy_dst[1] /= h

            if len(tmp_src_pts) < 500:
                zeros = np.zeros((500 - len(tmp_src_pts), 3))
                new_src_pts = np.vstack([new_src_pts, zeros])

                zeros = np.zeros((500 - len(tmp_src_pts), 2))
                new_dst_pts = np.vstack([new_dst_pts, zeros])

            new_src_pts = np.expand_dims(new_src_pts, axis=1)
            new_dst_pts = np.expand_dims(new_dst_pts, axis=1)

            if M is None:
                M = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
                new_src_pts = np.zeros((500, 1, 3))
                new_dst_pts = np.zeros((500, 1, 2))

            M[0][2] = M[0][2] / w
            M[1][2] = M[1][2] / h
        # print("M : ", M)
        # return set_t_1, set_t_0, M, st_0_0, flag
        data_st_0_0 = st_0_0

        if len(tp_good_kpt) > self.MIN_MATCH_COUNT:
            # print("matches are found - %d/%d" % (len(good_kpt), self.MIN_MATCH_COUNT))
            tp_src_pts = np.float32([kp0[m.queryIdx].pt for m in tp_good_kpt]).reshape(-1, 1, 2)
            tp_dst_pts = np.float32([kp2[m.trainIdx].pt for m in tp_good_kpt]).reshape(-1, 1, 2)

            tp_M, mask = cv2.findHomography(tp_src_pts, tp_dst_pts, cv2.RANSAC, 5.0)
            h, w = cv_old_frame.shape

            if tp_M is None:
                tp_M = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

            tp_M[0][2] = tp_M[0][2] / w
            tp_M[1][2] = tp_M[1][2] / h

        return set_t_1, set_t_0, M, data_st_0_0, new_src_pts, new_dst_pts, pt_num, tp_M
        # return set_t_1, set_t_0, M, st_0_0, 0, 0

    def __len__(self):
        """
        __len__() returns count of samples you have.
        """
        # 영상 path list의 사이즈, total 영상 수를 반환
        return len(self.image_path_list)  # of how many examples(images?) you have

    def load_image(self, image_index, flag, frame):

        if flag == 0:
            # stab
            img_type = self.folder_name[0]

            # if frame - 5 < 0:
            #     frame = frame + 5
            # len으로 해당 프레임까지 있나 먼저 확인해야 함

            # 영상 path list의 image_index번째를 뽑아서 image read -> img에 저장
            img_path = self.DB_path + img_type + self.folder_name[2] + '/' + self.name + '_' + str(frame) + '.jpg'

        elif flag == 1:
            # unstab
            img_type = self.folder_name[1]

            img_path = self.DB_path + img_type + self.folder_name[2] + '/' + self.name + '_' + str(frame) + '.jpg'

        img = skimage.io.imread(img_path)
        img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img, (288, 512))

        # img_scaled = preprocessing.scale(img)
        img_std = (img - np.mean(img)) / (np.max(img) - np.min(img))
        cv_img = (img * 255).astype(np.uint8)

        return img_std, img_path, cv_img

    def load_annotations(self, image_index):

        annotations = 0
        return annotations

    def _read_annotations(self):
        """

        """
        result = []
        cnt_list = {}
        for filepath in glob.glob(self.DB_path + self.folder_name[0] + self.folder_name[2] + '/*.jpg'):

            name, frame = self.find_param(filepath)

            if (filepath not in result) and (frame >= 33):
                result.append(filepath)

                if name not in cnt_list:
                    cnt_list[name] = 1
                else:
                    cnt_list[name] += 1

        sorted_cnt_list = collections.OrderedDict(sorted(cnt_list.items()))
        print("cnt_list : ", sorted_cnt_list)

        return result, cnt_list

    def find_param(self, filepath):
        filepath_split = filepath.split('/')
        filepath_reverse = filepath_split
        filepath_reverse.reverse()
        filename_extension = filepath_reverse[0]

        # print(filename_extension)
        filename_extension_split = filename_extension.split('.')
        filename_with_frame = filename_extension_split[0]
        filename_with_frame_split = filename_with_frame.split('_')
        filename = filename_with_frame_split[0]
        frame = filename_with_frame_split[1]

        return filename, int(frame)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)



# import pandas as pd
# import numpy as np
# from random import *
# import collections
#
# from matplotlib import pyplot as plt
# from PIL import Image
# import glob
#
# import cv2
# import sys
# import os
# import numpy as np
# import random
#
# import skimage.io
# import skimage.transform
# import skimage.color
# import skimage
# from sklearn import preprocessing
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# from torchvision import transforms
# from torch.utils.data.dataset import Dataset  # For custom datasets
# from operator import itemgetter
#
# """
# https://pytorch.org/docs/0.2.0/torchvision/transforms.html
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
# """
#
#
# class CSVDataset(Dataset):
#     """CSV dataset loader"""
#
#     def __init__(self, train_file, transform=None):
#         """
#         __init__() function is where the initial logic happens like reading a csv, assigning transforms etc.
#
#         Args:
#             csv_path (string): CSV file with training annotations
#             annotations (string): CSV file with class list
#             test_file (string, optional): CSV file with testing annotations
#
#         """
#
#         self.DB_path = train_file[0]
#         self.folder_name = train_file[1]
#
#         self.transform = transform
#
#         self.debug_name_check =[]
#
#
#
#         # csv with img_path, x1, y1, x2, y2, class_name
#         try:
#             self.image_path_list, self.image_cnt_len = self._read_annotations()
#             #print(self.image_path_list[0])
#             #random.shuffle(self.image_path_list)
#             #print(self.image_path_list[0])
#             #print("self.image_path_list", self.image_path_list)
#             #print("self.image_cnt_len", self.image_cnt_len)
#         except ValueError as e:
#             raise_from(ValueError('_read_annotations error: {}: {}'.format(self.DB_path, e)), None)
#
#         # keys() : 딕셔너리의 키를 모아서 dict_keys 객체 리턴
#         # image_data의 키를 리스트화해서 image_names에 저장, 영상 name list라고 보면 됨
#
#
#
#         #self.image_names = list(self.image_data.keys())
#
#
#     def _parse(self, value, function, fmt):
#         """
#         Parse a string into a value, and format a nice ValueError if it fails.
#         Returns `function(value)`.
#         Any `ValueError` raised is catched and a new `ValueError` is raised
#         with message `fmt.format(e)`, where `e` is the caught `ValueError`.
#         """
#         try:
#             return function(value)
#         except ValueError as e:
#             raise_from(ValueError(fmt.format(e)), None)
#
#     def _open_for_csv(self, path):
#         """
#         Open a file with flags suitable for csv.reader.
#         This is different for python2 it means with mode 'rb',
#         for python3 this means 'r' with "universal newlines".
#         """
#         if sys.version_info[0] < 3:
#             return open(path, 'rb')
#         else:
#             return open(path, 'r', newline='')
#
#
#     def load_classes(self, csv_reader):
#         result = {}
#
#
#         return result
#
#     def __getitem__(self, idx):
#         # stuff
#         """
#         __getitem__() function returns the data and labels.
#         This function is called from dataloader like this:
#         img, label = blah_blah_blah_Dataset.__getitem__(99)  # For 99th item
#         So, the idx parameter is the nth data/image (as tensor) you are going to return.
#
#         An important thing to note is that __getitem__() return a specific type for a single data point (like a tensor, numpy array etc.), otherwise, in the data loader you will get an error like:
#
#         TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>
#         """
#
#         self.name, self.frame = self.find_param(self.image_path_list[idx])  #190107 셔플 미리할 필요있음, 1회
#         # print("self.name(파일명), self.frame, idx : ", self.name, self.frame, idx)
#         # if self.name not in self.debug_name_check:
#         #     self.debug_name_check.append(self.name)
#
#         #print("debug_name_check : ", self.debug_name_check)
#         #self.frame_cnt = randint(5, self.image_cnt_len[self.name])
#         # if self.frame < 32: #190107 예외처리 제거, 데이터 로드할 때 32미만은 로드하지 않도록 수정
#         #     self.frame = self.frame + 5
#
#         #190107 인풋은 언스테디 프레임 I_t와 스테디 프레임 셋 S_t가 들어감 (학습시 S_t는 참값을 제공, 인퍼런스시 S_t는 앞서 추정된 결과를 누적해서 사용함)\
#         #190107 샴 네트웍 구조이므로 2번의 인풋셋을 입력으로 넣어줌
#         #190107 두번째에서는 I_t-1을 기준하여 인풋을 넣어줌
#         #190107 계산을 수행하고 각각의 로스를 구함
#         #190107 로스는 5종류가 존재함
#
#         #190107 1. stability loss
#         #190107 stability loss는 L_pixel과 L_feature로 이루어짐
#         #190107 L_pixel은 MSE와 같음
#         #190107 L_feature는 feature의 alignment 에러로 판단됨
#
#         #190107 2. Shape-preserving loss
#         #190107 L_intra와 L_inter로 구성됨
#         #190107 자세한 내용은 논문 확인 필요, 아직 안봤음
#
#         #190107 3. Temperoal loss
#         #190107 optical flow 기반해서 warping function을 찾고, 해당 warp 펑션으로 t-1과 t를 매칭시켜 MSE를 계산함
#
#         #190107 즉 MSE는 인풋 받아서 계산하는 함수 만들고
#         #190107 1번에서 MSE를 unsteady~staedy 사이에서 한번,
#         #190107 3번에서 t-1~t 사이에서 한번
#         #190107 구하도록 구현하면 좋음
#         stab_t_1_33, debug_path = self.load_image(idx, flag = 0, frame = self.frame-33)
#         #skimage.io.imshow(stab_img_32)
#
#         #skimage.io.show()
#
#         ##plt.show() # 이걸로 해도 됨
#         stab_t_1_17, debug_path = self.load_image(idx, flag = 0, frame = self.frame-17) # idx 번째에 해당하는 영상을 normalize 하여 불러옴
#         stab_t_1_9, debug_path  = self.load_image(idx, flag = 0, frame = self.frame-9)
#         stab_t_1_5, debug_path = self.load_image(idx, flag = 0, frame = self.frame-5)
#         stab_t_1_3, debug_path = self.load_image(idx, flag = 0, frame = self.frame-3)
#         stab_t_1_2, debug_path = self.load_image(idx, flag = 0, frame = self.frame-2)
#
#         unstab_t_1_1, debug_path = self.load_image(idx, flag = 1, frame = self.frame-1) # idx 번째에 해당하는 영상을 normalize 하여 불러옴
#
#         S_t_1 = {'I_33': stab_t_1_33,\
#         'I_17': stab_t_1_17,\
#         'I_9': stab_t_1_9,\
#         'I_5': stab_t_1_5,\
#         'I_3': stab_t_1_3,\
#         'I_2': stab_t_1_2,\
#         'I_un_1': unstab_t_1_1}
#
#         stab_t_0_32, debug_path = self.load_image(idx, flag = 0, frame = self.frame-32)
#         stab_t_0_16, debug_path = self.load_image(idx, flag = 0, frame = self.frame-16) # idx 번째에 해당하는 영상을 normalize 하여 불러옴
#         stab_t_0_8, debug_path = self.load_image(idx, flag = 0, frame = self.frame-8)
#         stab_t_0_4, debug_path = self.load_image(idx, flag = 0, frame = self.frame-4)
#         stab_t_0_2, debug_path = self.load_image(idx, flag = 0, frame = self.frame-2)
#         stab_t_0_1, debug_path = self.load_image(idx, flag = 0, frame = self.frame-1)
#
#         unstab_t_0_0, debug_path = self.load_image(idx, flag = 1, frame = self.frame-0) # idx 번째에 해당하는 영상을 normalize 하여 불러옴
#
#         stab_t_0_0, debug_path = self.load_image(idx, flag=0, frame=self.frame - 0)  # idx 번째에 해당하는 영상을 normalize 하여 불러옴
#
#
#         S_t_0 = {'I_32': stab_t_0_32,\
#         'I_16': stab_t_0_16,\
#         'I_8': stab_t_0_8,\
#         'I_4': stab_t_0_4,\
#         'I_2': stab_t_0_2,\
#         'I_1': stab_t_0_1,\
#         'I_un_0': unstab_t_0_0}
#         # if self.transform:
#         #     # If the transform variable is not empty
#         #     # then it applies the operations in the transforms with the order that it is created.
#         #     sample = self.transform(sample)
#
#         return (S_t_1, S_t_0, stab_t_0_0)
#
#     def __len__(self):
#         """
#         __len__() returns count of samples you have.
#         """
#         # 영상 path list의 사이즈, total 영상 수를 반환
#         return len(self.image_path_list) # of how many examples(images?) you have
#
#     def load_image(self, image_index, flag, frame):
#
#         res = []
#
#
#
#         if flag == 0:
#             #stab
#             img_type = self.folder_name[0]
#
#             # if frame - 5 < 0:
#             #     frame = frame + 5
#             #len으로 해당 프레임까지 있나 먼저 확인해야 함
#
#             # 영상 path list의 image_index번째를 뽑아서 image read -> img에 저장
#             img_path = self.DB_path + img_type + self.folder_name[2] +  '/' + self.name +  '_' + str(frame) + '.jpg'
#
#
#         elif flag == 1:
#             #unstab
#             img_type = self.folder_name[1]
#
#             img_path = self.DB_path + img_type + self.folder_name[2] +  '/' + self.name + '_' + str(frame) + '.jpg'
#
#
#
#         img = skimage.io.imread(img_path)
#
#         # if len(img.shape) == 2:
#         #     img = skimage.color.gray2rgb(img)
#
#         img = skimage.color.rgb2gray(img)
#         img = skimage.transform.resize(img, (288, 512))
#         #img_scaled = preprocessing.scale(img)
#         img = (img - np.mean(img)) / (np.max(img)-np.min(img))
#
#
#
#
#         #skimage.io.imshow(res[0])
#         #plt.show()
#
#         # Scaled data has zero mean and unit variance:
#         # >>>
#         # >>> X_scaled.mean(axis=0)
#         # array([0., 0., 0.])
#         # >>> X_scaled.std(axis=0)
#         # array([1., 1., 1.])
#
#         return img.astype(np.float32), img_path
#
#     def load_annotations(self, image_index):
#
#         annotations = 0
#         return annotations
#
#     def _read_annotations(self):
#         """
#
#         """
#         result = []
#         cnt_list = {}
#         for filepath in glob.glob(self.DB_path + self.folder_name[0] + self.folder_name[2] +  '/*.jpg'):
#
#             name, frame = self.find_param(filepath)
#
#
#             if (filepath not in result) and (frame >= 33):
#                 result.append(filepath)
#
#                 if name not in cnt_list:
#                     cnt_list[name] = 1
#                 else:
#                     cnt_list[name] += 1
#
#         sorted_cnt_list = collections.OrderedDict(sorted(cnt_list.items()))
#         print("cnt_list : ", sorted_cnt_list)
#
#
#
#
#
#
#
#
#         return result, cnt_list
#
#     def find_param(self, filepath):
#         filepath_split = filepath.split('/')
#         filepath_reverse = filepath_split
#         filepath_reverse.reverse()
#         filename_extension = filepath_reverse[0]
#
#         #print(filename_extension)
#         filename_extension_split = filename_extension.split('.')
#         filename_with_frame = filename_extension_split[0]
#         filename_with_frame_split = filename_with_frame.split('_')
#         filename = filename_with_frame_split[0]
#         frame = filename_with_frame_split[1]
#
#         return filename, int(frame)
#
#     def name_to_label(self, name):
#         return self.classes[name]
#
#     def label_to_name(self, label):
#         return self.labels[label]
#
#     def num_classes(self):
#         return max(self.classes.values()) + 1
#
#     def image_aspect_ratio(self, image_index):
#         image = Image.open(self.image_names[image_index])
#         return float(image.width) / float(image.height)
#
#
#
# #if __name__ == '__main__':
#     # Call the dataset
#     #custom_dataset = IMGRTDataset(...)
#

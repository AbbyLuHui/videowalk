import os
import numpy as np

import torch
import torch.utils.data as data
import random
import cv2



class GreaterList(data.Dataset):
    def __init__(self, filelist, clip_len, view=3, is_train=True, frame_gap=1, transform=None, random_clip=True):

        self.filelist = filelist
        self.clip_len = clip_len
        self.viewpoints = view
        self.is_train = is_train
        self.frame_gap = frame_gap

        self.random_clip = random_clip
        self.transform = transform
        self.RT = None
        self.K = None

        f = open(self.filelist, 'r')
        self.jpgfiles = []
        self.fnums = []

        for line in f:
            rows = line.split()
            jpgfile = rows[0]
            fnum = int(rows[1])

            self.jpgfiles.append(jpgfile)
            self.fnums.append(fnum)

        f.close()

    def __getitem__(self, index):
        index = index % len(self.jpgfiles)
        folder_path = self.jpgfiles[index]
        fnum = self.fnums[index]

        frame_gap = self.frame_gap
        startframe = 0

        readjust = False

        while fnum - self.clip_len * frame_gap < 0:
            frame_gap -= 1
            readjust = True

        if readjust:
            print('framegap adjusted to ', frame_gap, 'for', folder_path)

        diffnum = fnum - self.clip_len * frame_gap
        if self.random_clip:
            startframe = random.randint(0, diffnum)
        else:
            startframe = 0

        imgdirs= []
        img_files, depth_files, poses_RT, poses_K = {}, {}, {}, {}
        for view in range(self.viewpoints):
            imgdir = folder_path + "images_view" + str(view+1) + '/'
            imgdirs.append(imgdir)
            allfiles = os.listdir(imgdir)
            imgfiles = [f for f in allfiles if '_' not in f]
            depth = [f for f in allfiles if '_depth.png' in f]
            imgfiles.sort(key=lambda x: int(x.split('.')[0]))
            depth.sort(key=lambda x: int(x.split('_')[0]))
            img_files[view], depth_files[view] = imgfiles, depth

        if self.RT == None:
            for view in range(self.viewpoints):
                posedir = folder_path + 'poses_view' + str(view + 1) + '/'
                poses_RT[view] = np.load(posedir + 'camera_RT.npy')[1] #hack assume static camera view
                poses_K[view] = np.load(posedir + 'camera_K.npy')[1] #hack
            self.RT = poses_RT
            self.K = poses_K

        imgs, depths, poses = {i:[] for i in range(self.viewpoints)}, \
                              {i:[] for i in range(self.viewpoints)}, {}

        # reading video
        for i in range(self.clip_len):
            idx = int(startframe + i * frame_gap)
            for view in range(self.viewpoints):
                img_path = "%s/%s" % (imgdirs[view], img_files[view][idx])
                depth_path = "%s/%s" % (imgdirs[view], depth_files[view][idx])
                # BGR -> RGB!!!
                img = cv2.imread(img_path)[:, :, ::-1]  # .astype(np.float32)
                dep = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                imgs[view].append(img)
                depths[view].append(dep)

        for view in range(self.viewpoints):
            imgs[view] = np.stack(imgs[view])
            depths[view] = np.stack(depths[view])

        if self.transform is not None:
            for view in range(self.viewpoints):
                imgs[view] = self.transform(imgs[view])[0]
        def stack(dict):
            return np.stack(list(dict.values()))
        return stack(imgs), stack(depths), stack(self.RT), stack(self.K)

    def __len__(self):
        return len(self.jpgfiles)
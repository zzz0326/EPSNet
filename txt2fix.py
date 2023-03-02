import os
import glob
import cv2

import numpy as np

root_path = 'fixation txt path'

out_path = 'fixation map path'

video_path = 'video frame path'

list_video = [k.split('/')[-1] for k in glob.glob(os.path.join(video_path, '*'))]

video_frame_rate = np.zeros((216))

for i in range(len(list_video)):
    vidcap = cv2.VideoCapture(video_path + list_video[i])
    rate = round(vidcap.get(5))
    video_frame_rate[int(list_video[i][0:3])] = rate

list_file = [k.split('/')[-1] for k in glob.glob(os.path.join(root_path, '*'))]

for i in range(len(list_file)):
    txt_path = root_path + '/' + list_file[i]
    list_txt = [k.split('/')[-1] for k in glob.glob(os.path.join(txt_path, '*'))]
    video_id = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(txt_path, '*'))]
    for j in range(len(list_txt)):
        # frame head_x head_y eye_x eye_y
        txt = np.loadtxt(txt_path + '/' + list_txt[j], skiprows=0, delimiter=",", usecols=(1, 3, 4, 6, 7))
        txt[:, 1] = txt[:, 1] * 320 - 1
        txt[:, 3] = txt[:, 3] * 320 - 1
        txt[:, 2] = txt[:, 2] * 160 - 1
        txt[:, 4] = txt[:, 4] * 160 - 1
        txt = txt.astype(int)
        count = 0
        map_path = out_path + video_id[j] + 'video_' + str(count) + '.png'
        if os.path.exists(map_path):
            map = cv2.imread(map_path, 0).astype(int)
        else:
            map = np.zeros((160, 320, 1), dtype=int)
        for k in range(txt.shape[0]):
            if k % video_frame_rate[int(video_id[j])] == 0 and k > 0:
                cv2.imwrite(map_path, map)
                count += 1
                map_path = out_path + video_id[j] + 'video_' + str(count) + '.png'
                if os.path.exists(map_path):

                    map = cv2.imread(map_path, 0).astype(int)
                else:
                    map = np.zeros((160, 320, 1), dtype=int)
            map[txt[k, 2], txt[k, 1]] += 255
            map[txt[k, 4], txt[k, 3]] += 255

        cv2.imwrite(map_path, map)

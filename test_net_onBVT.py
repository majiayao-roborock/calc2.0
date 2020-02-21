
#!/usr/bin/env python

import cv2
import numpy as np
import code
import time
import os
import scipy.io
import csv
import datetime
#import keras
#from keras_helpers import *
import pickle
import heapq
import copy

def read_imgs_txt(file_path,freq):
    images_file = open(file_path,'r')
    try:
        images_path_lines = images_file.readlines()
    finally:
        images_file.close()
    imgs = []
    index = 0
    for image_path in images_path_lines:
        if index % freq != 0:
            continue
        print( 'index: ',index)
        image_path = image_path.rstrip('\n')
        imgs.append(image_path)
        index = index + 1
    return imgs

def read_imgs(file_path,freq):
    images_path_lines = csv.reader(open(file_path + 'data.csv','rb'))
    imgs = []
    imgs_timestep = []
    index = -1
    for image_path in images_path_lines:
        index = index + 1
        if index == 0:
            continue
        if index % freq != 0:
            continue
        imgs_timestep.append(image_path[0])
        image_path_full = file_path + 'data/' + image_path[1]
        #print 'image_path full:',image_path
        imgs.append(image_path_full)
    return imgs,imgs_timestep


def find_pose(poses_path,timestep):
    pose_lines = csv.reader(open(poses_path ,'r'))
    index = -1
    for pose in pose_lines:
        index = index + 1
        if index == 0:
            continue
        if pose[0] < timestep:
            continue
        else:
            #print ' ts:',pose[0] ,' ',timestep
            #print 'pose',pose
            pose_x = float(pose[1])
            pose_y = float(pose[2])
            theta = float(pose[3])
            pose_x_y = np.array([pose_x,pose_y])
            #print 'res:',res
            return pose_x_y,theta

def pose_dist(query_id,query_result):
    query_pose,query_theta = find_pose(query_poses_path,query_ts[query_id])
    base_pose,base_theta = find_pose(base_poses_path,base_ts[query_result])
    dist = np.sqrt(np.sum(np.square(query_pose-base_pose)))
    phase = abs(query_theta - base_theta)
    return dist,phase

def check_match_pair(query_id,query_result):
    dist,phase = pose_dist(query_id,query_result)
    print ('dist,phase',dist,phase)
    if dist < max_dist:# and phase < max_phase:
        return True
    else:
        # cv2.imshow( 'query_image',cv2.imread(query_imgs[query_id]))
        # cv2.imshow( 'match_image',cv2.imread(base_imgs[query_result[0]]))
        # cv2.waitKey(500)
        # judge = -1
        # while judge !=0 and  judge!=1:
        #     judge = input("match: 1; not match 0:  ")
        #     print 'judge',judge
        # if judge == 1:
        #     print 'accept!'
        #     return True
        # else:
        #     return False
        return False

def find_match(query_ts,query_desc,base_desc,query_imgs,base_imgs):
    query_id = -1
    true_match = 0
    avg_time = 0.0
    all_tag = []
    all_score = []
    for query in query_desc:
        query_id = query_id + 1
        print ('query once',query_id)
        #cv2.imshow( 'query_image',cv2.imread(query_imgs[query_id]))
        #cv2.waitKey(0)
        score = []
        query_result = []
        for base in base_desc:
            onescore = query.dot(base.transpose())
            #print ('onescore: ',onescore)
            score.append(onescore)

        max_score = heapq.nlargest(topk,score)
        print ('---maxscore: ',max_score)
        max_score_index = map(score.index,max_score)
        query_result.extend(max_score_index)

        check_pair = True
        i = 0
        all_score.append(float(max_score[0]))
        for query_top in query_result:
            if check_pair and check_match_pair(query_id,query_top):
                true_match = true_match + 1
                all_tag.append(1)


                filepath = './res_true/'+ query_ts[query_id]
                cv2.imwrite(filepath + '.png',cv2.imread(query_imgs[query_id]))
                cv2.imwrite(filepath + '-' + str(i) + '-t-'+ str(max_score[i]) +'-' + str(base_ts[query_top]) +'.png',cv2.imread(base_imgs[query_top]))

            else:
                all_tag.append(0)

                filepath = './res_far/'+ query_ts[query_id]
                cv2.imwrite(filepath + '.png',cv2.imread(query_imgs[query_id]))
                cv2.imwrite(filepath + '-' + str(i) + '-f-'+ str(max_score[i]) +'-' + str(base_ts[query_top]) +'.png',cv2.imread(base_imgs[query_top]))
            i = i + 1
    avg_time = avg_time/float(query_id+1)
    print( 'avg find match time' ,avg_time,' ms, for',query_id+1,' pics')
    print ('len all_tag',len(all_tag))
    print ('len all score',len(all_score))
    print ('all_tag',all_tag,end=' ')
    print ('all_score',all_score,end=' ')
    return [query_id+1,true_match]


base_imgs_path = '/home/jiayao/catkin_ws_docker/netvlad/dataset/base/2019082701/TansoV/cam0/' # on docker
query_imgs_path = '/home/jiayao/catkin_ws_docker/netvlad/dataset/query/2019082702/TansoV/cam0/' # on docker
base_poses_path = "/home/jiayao/catkin_ws_docker/netvlad/dataset/base/2019082701/TansoV/slampose/data.csv";
query_poses_path = "/home/jiayao/catkin_ws_docker/netvlad/dataset/query/2019082702/TansoV/slampose/data.csv";
topk= 1
max_dist = 2
max_phase = 2

if __name__ == "__main__":
    #save = False  ### use pickle to save variable ###

    base_imgs = pickle.load(open('./res/base_imgs.txt','rb'))
    base_ts = pickle.load(open('./res/base_ts.txt','rb'))
    base_desc = pickle.load(open('./res/base_img_desc_calc.txt','rb'))

    query_imgs = pickle.load(open('./res/query_imgs.txt','rb'))
    query_ts = pickle.load(open('./res/query_ts.txt','rb'))
    query_desc = pickle.load(open('./res/query_img_desc_calc.txt','rb'))


    print( 'len(base_desc),len(query_desc)',len(base_desc),len(query_desc))
    print ('desc shape= ' ,base_desc[0].shape)

    [all_query,true_match]= find_match(query_ts,query_desc,base_desc,query_imgs,base_imgs)

    print ('find match:',all_query)
    print( 'true match:',true_match)
    print ('false positive: ',(100.0*true_match)/float(all_query*topk),"%")
#print 'match_dic',match_dict
#np.savetxt(output_path, mem_images,delimiter= " ")
#np.savetxt(output_path, test, fmt="%.2f,%.2f",delimiter= " ")




"""
@author: sotiris
"""

from mmdet.apis import init_detector, inference_detector

import os
import pickle
import cv2
import numpy as np
import argparse
from utilities import match

def infer_model(img_paths, settings, root_dir, thresholds):
    # build the model from a config file and a checkpoint file
    model = init_detector(settings['config_path'], settings['model_path'], device='cuda:0')
    
    # detect objects from a single image
    detections_vehicle = {}
    detections_traffic_light = {}
    detections_traffic_sign = {}
    for frame in annotated_frames_paths:
        frame_detections_vehicle = []
        frame_detections_traffic_light = []
        frame_detections_traffic_sign = []
        dets = inference_detector(model, frame)
        for bbox in dets[2]:
            if bbox[4] >= thresholds['thresh_car']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'vehicle'
                frame_detections_vehicle.append(box_data)
        for bbox in dets[3]:
            if bbox[4] >= thresholds['thresh_truck']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'truck'
                frame_detections_vehicle.append(box_data)
        for bbox in dets[4]:
            if bbox[4] >= thresholds['thresh_bus']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'bus'
                frame_detections_vehicle.append(box_data)
        for bbox in dets[6]:
            if bbox[4] >= thresholds['thresh_motorcycle']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'motorcycle'
                frame_detections_vehicle.append(box_data)
        for bbox in dets[8]:
            if bbox[4] >= thresholds['thresh_traffic_light']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'traffic_light'
                frame_detections_traffic_light.append(box_data)
        for bbox in dets[9]:
            if bbox[4] >= thresholds['thresh_traffic_sign']:
                box_data = {}
                box_data['box_points'] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                box_data['confidence'] = float(bbox[4])
                box_data['class'] = 'traffic_sign'
                frame_detections_traffic_sign.append(box_data)
        
        detections_vehicle[frame.split('/')[-1]] = frame_detections_vehicle 
        detections_traffic_light[frame.split('/')[-1]] = frame_detections_traffic_light
        detections_traffic_sign[frame.split('/')[-1]] = frame_detections_traffic_sign
            
    return detections_vehicle, detections_traffic_light, detections_traffic_sign

def make_inputs(anno_path, frames_dir):
    try: # exception handling wrong path to annotation file
        with open(anno_path, 'rb') as f:
            annotations = pickle.load(f)
    except:
        print('annotations path not valid')
        
    annotated_frames_paths = []
    bboxes_vehicle = {}
    bboxes_traffic_light = {}
    bboxes_traffic_sign = {}
    for key in annotations:
        annotated_frames_paths += [os.path.join(frames_dir,key)]        
        img = cv2.imread(annotated_frames_paths[-1]) # read image to check if corrupted
        if img is None:
            print("corrupted image file")
            annotated_frames_paths.pop()
        else:
            frame_bboxes_vehicle = []
            frame_bboxes_traffic_light = []
            frame_bboxes_traffic_sign = []
            for bbox in annotations[key]['boxes']:
                if bbox['class'] == 'Vehicle':
                    frame_bboxes_vehicle.append(np.array([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]))
                if bbox['class'] == 'Traffic Light':
                    frame_bboxes_traffic_light.append(np.array([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]))
                if bbox['class'] == 'Traffic Sign':
                    frame_bboxes_traffic_sign.append(np.array([bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']]))                    
            bboxes_vehicle[key] = frame_bboxes_vehicle
            bboxes_traffic_light[key] = frame_bboxes_traffic_light
            bboxes_traffic_sign[key] = frame_bboxes_traffic_sign
                                
    return annotated_frames_paths, bboxes_vehicle, bboxes_traffic_light, bboxes_traffic_sign

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='object detection of selected images')
    parser.add_argument('--anno_path',
                       dest='annotation_path',
                       type=str,
                       required=True,
                       help="path to annotation directory"
                       )
    parser.add_argument('--frames_dir',
                        dest='frames_dir',
                        type=str,
                        required=True,
                        help="path to images directory"
                        )
    parser.add_argument('--config',
                        nargs=2,
                        dest='cfg_list',
                        required=True,
                        help="a list of paths with first path specifying the model file and the second path the config file"
                        ) 
    parser.add_argument('--out_dir',
                      dest='out_dir',
                      type=str,
                      required=False,
                      default=os.getcwd(),
                      help="path to the output directory containing the pickle files"
                      )  
    parser.add_argument('--thresh_car',
                        dest='thresh_car',
                        type=float,
                        required=False,
                        default=0.95,
                        help="car detection threshold"
                        )
    parser.add_argument('--thresh_bus',
                        dest='thresh_bus',
                        type=float,
                        required=False,
                        default=0.95,
                        help="bus detection threshold"
                        )
    parser.add_argument('--thresh_truck',
                        dest='thresh_truck',
                        type=float,
                        required=False,
                        default=0.95,
                        help="truck detection threshold"
                        )
    parser.add_argument('--thresh_motorcycle',
                        dest='thresh_motorcycle',
                        type=float,
                        required=False,
                        default=0.95,
                        help="motorcycle detection threshold"
                        )
    parser.add_argument('--thresh_traffic_light',
                        dest='thresh_traffic_light',
                        type=float,
                        required=False,
                        default=0.5,
                        help="traffic light detection threshold"
                        )
    parser.add_argument('--thresh_traffic_sign',
                        dest='thresh_traffic_sign',
                        type=float,
                        required=False,
                        default=0.75,
                        help="traffic sign detection threshold"
                        )
    
    args = parser.parse_args()

    annotated_frames_paths, bboxes_vehicle, bboxes_traffic_light, bboxes_traffic_sign = make_inputs(args.annotation_path, args.frames_dir)

    settings = {"model_path": args.cfg_list[0],
                "config_path": args.cfg_list[1],
                }
    for i in range(len(args.cfg_list[0])):
        assert os.path.isfile(settings['model_path']), "Not a valid model file %s"\
            % (settings['model_path'])
        assert os.path.isfile(settings['config_path']), "Not a valid model file %s"\
            % (settings['config_path'])    
    thresholds = {
        'thresh_car': args.thresh_car,
        'thresh_bus': args.thresh_bus,
        'thresh_truck': args.thresh_truck,
        'thresh_motorcycle': args.thresh_motorcycle,
        'thresh_traffic_light': args.thresh_traffic_light,
        'thresh_traffic_sign': args.thresh_traffic_sign
            }

    detections_vehicle, detections_traffic_light, detections_traffic_sign = infer_model(annotated_frames_paths, settings, thresholds)

    mdict_list_vehicle = []
    mdict_list_traffic_light = []
    mdict_list_traffic_sign = []
    for i in range(len(annotated_frames_paths)):    
        mdict_vehicle = dict( )
        mdict_traffic_light = dict()
        mdict_traffic_sign = dict()
    
        mdict_vehicle["file_path"] = annotated_frames_paths[i]
        mdict_traffic_light["file_path"] = annotated_frames_paths[i]
        mdict_traffic_sign["file_path"] = annotated_frames_paths[i]
    
        mdict_vehicle["matches"] = match(detections_vehicle[annotated_frames_paths[i].split('/')[-1]], bboxes_vehicle[annotated_frames_paths[i].split('/')[-1]])
        mdict_traffic_light["matches"] = match(detections_traffic_light[annotated_frames_paths[i].split('/')[-1]], bboxes_traffic_light[annotated_frames_paths[i].split('/')[-1]])
        mdict_traffic_sign["matches"] = match(detections_traffic_sign[annotated_frames_paths[i].split('/')[-1]], bboxes_traffic_sign[annotated_frames_paths[i].split('/')[-1]])
    
        mdict_vehicle["det"] = detections_vehicle[annotated_frames_paths[i].split('/')[-1]]
        mdict_traffic_light["det"] = detections_traffic_light[annotated_frames_paths[i].split('/')[-1]]
        mdict_traffic_sign["det"] = detections_traffic_sign[annotated_frames_paths[i].split('/')[-1]]
        
        mdict_vehicle["gt"] = [dict({"bbox": bbox, "class": r"Vehicle"}) for bbox in bboxes_vehicle[annotated_frames_paths[i].split('/')[-1]]]
        mdict_traffic_light["gt"] = [dict({"bbox": bbox, "class": r"Traffic Light"}) for bbox in bboxes_traffic_light[annotated_frames_paths[i].split('/')[-1]]]
        mdict_traffic_sign["gt"] = [dict({"bbox": bbox, "class": r"Traffic Sign"}) for bbox in bboxes_traffic_sign[annotated_frames_paths[i].split('/')[-1]]]
    
        mdict_list_vehicle.append(mdict_vehicle)
        mdict_list_traffic_light.append(mdict_traffic_light)
        mdict_list_traffic_sign.append(mdict_traffic_sign)

    # save mdicts as pickle files
    pickle.dump(mdict_list_vehicle, open(os.path.join(args.out_dir,'mdict_list_vehicle.pkl'), 'wb'))
    pickle.dump(mdict_list_traffic_light, open(os.path.join(args.out_dir,'mdict_list_traffic_light.pkl'), 'wb'))
    pickle.dump(mdict_list_traffic_sign, open(os.path.join(args.out_dir, 'mdict_list_traffic_sign.pkl'), 'wb'))

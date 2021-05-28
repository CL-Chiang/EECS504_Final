import sys
sys.path.insert(0, "./Object_Detection/YOLOv3/")
from Object_Detection.detect import YOLO_V3
from Object_Detection.detect import Detection_Interface
import cv2
import os.path
import orbslam2
import time
import argparse


def main(vocab_path, settings_path, sequence_path, dl):

    left_filenames, right_filenames, timestamps = load_images(sequence_path)
    num_images = len(timestamps)

    slam = orbslam2.System(vocab_path, settings_path, orbslam2.Sensor.STEREO)
    slam.set_use_viewer(True)
    slam.initialize()

    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence ...')
    print('Images in the sequence: {0}'.format(num_images))

    for idx in range(num_images):
        left_image = cv2.imread(left_filenames[idx], cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(right_filenames[idx], cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if left_image is None:
            print("failed to load image at {0}".format(left_filenames[idx]))
            return 1
        if right_image is None:
            print("failed to load image at {0}".format(right_filenames[idx]))
            return 1

        image_dl = cv2.imread(left_filenames[idx])
        height, width, depth = image_dl.shape
        all_detections = dl.detect(image_dl[:, :width//2, :])
        

        od = orbslam2.OD()

        for i, detections in enumerate(all_detections):
            if detections is not None:  
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if cls_conf < 0.8:
                        continue
                    cls_pred = int(cls_pred)
                    # For COCO, pedstrian=0, car=2, truck=7, and ignore other classes
                    if i == 0:
                        if cls_pred == 0:
                            cls_pred = 43
                        elif cls_pred == 2:
                            cls_pred = 44
                        elif cls_pred == 7:
                            cls_pred = 45
                        else:
                            continue
                    # print(i, cls_pred)
                    od.set(int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1), cls_pred)
        all_detections = dl.detect(image_dl[:, width//2:, :])
        for i, detections in enumerate(all_detections):
            if detections is not None:  
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if cls_conf < 0.8:
                        continue
                    cls_pred = int(cls_pred)
                    # For COCO, pedstrian=0, car=2, truck=7, and ignore other classes
                    if i == 0:
                        if cls_pred == 0:
                            cls_pred = 43
                        elif cls_pred == 2:
                            cls_pred = 44
                        elif cls_pred == 7:
                            cls_pred = 45
                        else:
                            continue
                    # print(i, cls_pred)
                    od.set(int(x1)+width//2, int(y1), int(x2)-int(x1), int(y2)-int(y1), cls_pred)




        tframe = timestamps[idx]

        t1 = time.time()
        # slam.process_image_mono(left_image, a, tframe)
        # set the same results for 2 cameras temp
        slam.process_image_stereo(left_image, right_image, od, od, tframe)
        t2 = time.time()

        ttrack = t2 - t1
        times_track[idx] = ttrack

        t = 0
        if idx < num_images - 1:
            t = timestamps[idx + 1] - tframe
        elif idx > 0:
            t = tframe - timestamps[idx - 1]

        if ttrack < t:
            time.sleep(t - ttrack)

    #save_trajectory(slam.get_trajectory_points(), 'trajectory.txt')

    slam.shutdown()

    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    return 0


def load_images(path_to_sequence):
    timestamps = []
    with open(os.path.join(path_to_sequence, 'times.txt')) as times_file:
        for line in times_file:
            if len(line) > 0:
                timestamps.append(float(line))

    raw = 1
    if raw == 0:
    
        return [
            os.path.join(path_to_sequence, 'image_0', "{0:06}.png".format(idx))
            for idx in range(len(timestamps))
        ], [
            os.path.join(path_to_sequence, 'image_1', "{0:06}.png".format(idx))
            for idx in range(len(timestamps))
        ], timestamps
    
    else:
        return [
            os.path.join(path_to_sequence, 'image_02/data/', "{0:010}.png".format(idx))
            for idx in range(len(timestamps))
        ], [
            os.path.join(path_to_sequence, 'image_03/data/', "{0:010}.png".format(idx))
            for idx in range(len(timestamps))
        ], timestamps
 
 



def save_trajectory(trajectory, filename):
    with open(filename, 'w') as traj_file:
        traj_file.writelines('{time} {r00} {r01} {r02} {t0} {r10} {r11} {r12} {t1} {r20} {r21} {r22} {t2}\n'.format(
            time=repr(stamp),
            r00=repr(r00),
            r01=repr(r01),
            r02=repr(r02),
            t0=repr(t0),
            r10=repr(r10),
            r11=repr(r11),
            r12=repr(r12),
            t1=repr(t1),
            r20=repr(r20),
            r21=repr(r21),
            r22=repr(r22),
            t2=repr(t2)
        ) for stamp, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2 in trajectory)

def create_yolo_opt(task=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./Object_Detection/YOLOv3/data/samples", help="path to dataset")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    if task == "COCO":
        parser.add_argument("--model_def", type=str, default="./Object_Detection/YOLOv3/config/yolov3.cfg", help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="./Object_Detection/YOLOv3/weights/yolov3.weights", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="./Object_Detection/YOLOv3/data/coco.names", help="path to class label file")
    elif task == "GTSDB":
        parser.add_argument("--model_def", type=str, default="./Object_Detection/YOLOv3/config/yolov3-GTSDB.cfg", help="path to model definition file")
        parser.add_argument("--weights_path", type=str, default="./Object_Detection/YOLOv3/weights/yolov3_ckpt_1900.pth", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="./Object_Detection/YOLOv3/data/custom/classes.names", help="path to class label file")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    print("EECS504 Project")
    yolo_COCO = YOLO_V3(create_yolo_opt("COCO"))
    yolo_GTSDB = YOLO_V3(create_yolo_opt("GTSDB"))
    dl = Detection_Interface([yolo_COCO, yolo_GTSDB])

    orb_txt = "./SLAM/SLAM_CPP/Vocabulary/ORBvoc.txt"
    yaml = "./SLAM/SLAM_CPP/Examples/Stereo/KITTI00-02.yaml"
    # dataset = "/home/parallels/Desktop/Parallels Shared Folders/Home/dataset/sequences/06/"
    dataset = "./data/2011_09_26-5/2011_09_26_drive_0005_sync"

    main(orb_txt, yaml, dataset, dl)







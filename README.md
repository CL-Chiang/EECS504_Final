EECS504 Final Project: Object Detection with SLAM
====

### Prerequesities

- Ubuntu 18
- Python3
- OpenCV 3.4
- curl
- libboost-python1.65-dev
- libpython3.7m
- Pangolin
- Pandas
- numpy

### Preparation

#### Uncompress ORB Vocab

```
cd SLAM/SLAM_CPP/Vocabulary/
tar xvf ORBvoc.txt.tar.gz
cd -
```


#### Get Yolov3 model

```
cd Object_Detection/YOLOv3/weights
./get_yolov3_model.sh # it may take a while
cd -
```

#### Get orbslam dynamic link lib

```
cd ORB_SLAM_LIB
./get_orb_slam_lib.sh # it may take a while
sudo cp *.so /usr/local/lib/. 
export PYTHONPATH=`pwd`
cd -
```

#### Get example data

```
cd ORB_SLAM_LIB
./get_data.sh # it may take a while
tar xvf 2011_09_26-5.tar.xz 
cd -
```
#### Execute
```
python3 main.py
```




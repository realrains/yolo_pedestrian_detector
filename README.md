# yolo_pedestrian_detector

### Requirement

1. tensorflow C++ API
  * (Build only C++ API) [tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc)  


```
git clone git clone https://github.com/FloopCZ/tensorflow_cc.git
cd tensorflow_cc/tensorflow_cc
mkidr build && cd build
cmake ..
make
sudo make install
```

2. other libraries  


```
sudo apt-get install build-essential curl git cmake unzip autoconf autogen libtool mlocate zlib1g-dev \
python python3-numpy python3-dev python3-pip python3-wheel
```

### download pretrained model

[https://drive.google.com/file/d/12fQhOYvfA833tD_xZ9c8cvcquV87EXiA/view?usp=sharing](https://drive.google.com/file/d/12fQhOYvfA833tD_xZ9c8cvcquV87EXiA/view?usp=sharing)

### build

```
cd yolo_pedestrian_detector
cmake CMakeLists.txt
make
```

### test demo

```
./human_detect ./frozen_graph.pb ./pedestrian_2.jpg
```

# ICPCUDA-pod
ICPCUDA-pod

Uses ICPCUDA to estimate motion (and velocity)

Depends on bot_core types, kinect types and Consise args

Compile as follows:

:

   cd ICPCUDA-pod
   git submodule update --init --recursive
   make

You might need to adjust you detected cuda architecture:

:

   cd pod-build
   ccmake .

Then set CUDA_ARCH_BIN to 30. Then continue to make with one core :

:
   cd ICPCUDA-pod
   make -j 1


Test program:

:
   ICP  ~/logs/kinect/rgbd_dataset_freiburg1_desk/ -v

LCM applications:

:
   se-icpcuda


You might need to run with sudo permisions on first run



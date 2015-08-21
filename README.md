# ICPCUDA-pod
ICPCUDA-pod

Might need to addjust

:

   cd pod-build
   ccmake .

Then set CUDA_ARCH_BIN to 30

Then make with one core:

:
   cd ..
   make -j 1


Test program:

ICP  ~/logs/kinect/rgbd_dataset_freiburg1_desk/ -v

Mihgt need to run with sudo permisions on first run

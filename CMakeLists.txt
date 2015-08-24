cmake_minimum_required(VERSION 2.6.0)

# pull in the pods macros. See cmake/pods.cmake for documentation
set(POD_NAME ICPCUDA-app)
include(cmake/pods.cmake)

# automatically build LCM types.  This also defines a number of CMake
# variables, see cmake/lcmtypes.cmake for details
include(cmake/lcmtypes.cmake)
lcmtypes_build()

#add_subdirectory(ICPCUDA/src)
#SET(srcs ICPCUDA/src/ICP.cpp ICPCUDA/src/ICPOdometry.cpp ICPCUDA/src/ICPSlowdometry.cpp)
#SET(cuda ICPCUDA/src/Cuda/estimate_combined.cu ICPCUDA/src/Cuda/pyrdown.cu ICPCUDA/src/Cuda/icp.cu)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread filesystem system REQUIRED)
find_package(CUDA REQUIRED)

#remove this as soon as eigen is shipped with FindEigen.cmake
  get_filename_component(EIGEN_ROOT "/usr/include/eigen3" PATH)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_EIGEN eigen3)
  endif(PKG_CONFIG_FOUND)
  find_path(EIGEN_INCLUDE_DIRS Eigen/Core
    HINTS ${PC_EIGEN_INCLUDEDIR} ${PC_EIGEN_INCLUDE_DIRS} 
          "${EIGEN_ROOT}" "$ENV{EIGEN_ROOT}"
    PATHS "$ENV{PROGRAMFILES}/Eigen 3.0.0" "$ENV{PROGRAMW6432}/Eigen 3.0.0"
          "$ENV{PROGRAMFILES}/Eigen" "$ENV{PROGRAMW6432}/Eigen"   
    PATH_SUFFIXES eigen3 include/eigen3 include)
  find_package_handle_standard_args(eigen DEFAULT_MSG EIGEN_INCLUDE_DIRS)
  set(EIGEN_DEFINITIONS ${EIGEN_DEFINITIONS} -DEIGEN_USE_NEW_STDVECTOR 
      -DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)
      
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})

#file(GLOB srcs *.cpp)
#file(GLOB cuda Cuda/*.cu)
#file(GLOB containers Cuda/containers/*.cpp)
file(GLOB srcs ICPCUDA/src/ICPOdometry.cpp ICPCUDA/src/ICPSlowdometry.cpp)
file(GLOB cuda ICPCUDA/src/Cuda/*.cu)
file(GLOB containers ICPCUDA/src/Cuda/containers/*.cpp)

set(CUDA_ARCH_BIN "35" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(ICPCUDA/src/CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})


add_library(icpcuda SHARED ICPCUDA/src/ICPSlowdometry.cpp ICPCUDA/src/ICPOdometry.cpp ${cuda} ${containers} ICPCUDA/src/Cuda/icp.cu ICPCUDA/src/Cuda/estimate_combined.cu ICPCUDA/src/Cuda/pyrdown.cu)
pods_use_pkg_config_packages(icpcuda eigen3)

file(GLOB src_hpp ICPCUDA/src/*.h)
file(GLOB cuda_hpp ICPCUDA/src/Cuda/*.h)
file(GLOB containers_hpp ICPCUDA/src/Cuda/containers/*.hpp)

set_target_properties(icpcuda PROPERTIES SOVERSION 1)
pods_install_libraries(icpcuda)
pods_install_headers( ${src_hpp}  DESTINATION icpcuda)
pods_install_headers( ${cuda_hpp}  DESTINATION icpcuda/Cuda)
pods_install_headers( ${containers_hpp}  DESTINATION icpcuda/Cuda/containers)
pods_install_pkg_config_file(icpcuda
    LIBS -licpcuda
    REQUIRES 
    VERSION 0.0.1)


add_executable(ICP  ICPCUDA/src/ICP.cpp ${srcs} ${cuda} ${cuda_objs} ${containers})
target_link_libraries(ICP ${Boost_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${CUDA_LIBRARIES})
pods_install_executables(ICP)

add_executable(icpcuda-app-original src/icpcuda-app/icpcuda-app-original.cpp ${srcs} ${cuda} ${cuda_objs} ${containers})
target_link_libraries(icpcuda-app-original ${Boost_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${CUDA_LIBRARIES})
pods_install_executables(icpcuda-app-original)

add_executable(se-icpcuda-simple src/icpcuda-app/icpcuda-app-simple.cpp ${srcs} ${cuda} ${cuda_objs} ${containers})
target_link_libraries(se-icpcuda-simple ${Boost_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${CUDA_LIBRARIES} z)
pods_use_pkg_config_packages(se-icpcuda-simple eigen3 lcm lcmtypes_kinect lcmtypes_bot2-core)
pods_install_executables(se-icpcuda-simple)

add_executable(se-icpcuda src/icpcuda-app/icpcuda-app.cpp ${srcs} ${cuda} ${cuda_objs} ${containers})
target_link_libraries(se-icpcuda ${Boost_LIBRARIES} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${CUDA_LIBRARIES} z)
pods_use_pkg_config_packages(se-icpcuda eigen3 lcm lcmtypes_kinect lcmtypes_bot2-core bot2-frames)
pods_install_executables(se-icpcuda)

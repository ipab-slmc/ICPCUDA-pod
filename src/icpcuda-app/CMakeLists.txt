set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated-declarations")


add_definitions(-Wall )
#add_definitions(-Wall -std=gnu99) # .... compiler warning

##############################################
# Basic Test Program
#add_executable (se-icpcuda icpcuda-app.cpp)
#target_link_libraries(se-icpcuda boost_system)
#pods_use_pkg_config_packages(se-icpcuda icpcuda
#  lcm
#  lcmtypes_bot2-core)
#pods_install_executables(se-icpcuda)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread filesystem system REQUIRED)
find_package(CUDA REQUIRED)
find_package(ZLIB REQUIRED)

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


set(CUDA_ARCH_BIN "35" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(../../cmake/CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})


add_executable (se-icpcuda-original icpcuda-app-original.cpp)
target_link_libraries(se-icpcuda-original boost_system ${OpenCV_LIBS} ${CUDA_LIBRARIES} /usr/lib/x86_64-linux-gnu/libcuda.so
                      /usr/lib/x86_64-linux-gnu/libcudart.so /usr/lib/x86_64-linux-gnu/libicudata.so   ${ZLIB_LIBRARIES}
                      /home/mfallon/otherprojects/pronto-distro/build/lib/libicpcuda.so)
pods_use_pkg_config_packages(se-icpcuda-original 
  lcm 
  lcmtypes_bot2-core
  )
pods_install_executables(se-icpcuda-original)




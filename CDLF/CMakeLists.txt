
if (${Use_GPU})
   # set(CMAKE_CUDA_COMPILER /opt/apps/cuda/9.1.85.2/bin/nvcc)

   # this must correctly config, otherwise GPU can not work correctly
   #set(CMAKE_CUDA_FLAGS -arch=sm_70)   #Nvidia Titan V in COE_GPU
   set(CMAKE_CUDA_FLAGS -arch=sm_60)   #Nvidia Tesla P100 in UI_GPU
endif(${Use_GPU})

#[[ methodds of getting  GPU informaiton
  use lspci | grep -i nvidia to get GPU information
  use nvidia-smi -q to get more detailed information of GPU
  use cat  /proc/driver/nvidia/gpus/0000:3b:00.0/information to get model number

[hxie1@argon-itf-ca38-07 0000:3b:00.0]$ cat /proc/driver/nvidia/gpus/0000:3b:00.0/information
Model: 		 TITAN V
IRQ:   		 136
GPU UUID: 	 GPU-a6cba209-8da6-6676-9415-e364f1035a3a
Video BIOS: 	 88.00.41.00.18
Bus Type: 	 PCIe
DMA Size: 	 47 bits
DMA Mask: 	 0x7fffffffffff
Bus Location: 	 0000:3b:00.0
Device Minor: 	 0

in this case, Titan V has a compute capability 7.0

]]


message("Config StatsLab Notes:")
message("0  Courtesy of Keith O'Hara (https://github.com/kthohr/stats) for StatsLab")
message("1  download code from: https://github.com/kthohr/stats into your intall directory")
message("2  in Clion File/Setting/Build/CMake/CMakeOptions config -DSTATS_INCLUDE_DIR=The/Full/Path/Of/Stats/Include/Dir ")
message("*************************************")

#set(STATS_INCLUDE_DIR /Users/hxie1/Projects/stats/include)
include_directories(${STATS_INCLUDE_DIR})

set(CDLF_CXX_Src  Layer.cpp FCLayer.cpp InputLayer.cpp LossLayer.cpp FeedForwardNet.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.h Tools.cpp
        ConvolutionLayer.cpp BiasLayer.cpp SoftmaxLayer.cpp MaxPoolingLayer.cpp
        IdentityLayer.cpp MergerLayer.cpp BranchLayer.cpp
        SigmoidLayer.cpp  CrossEntropyLoss.cpp VectorizationLayer.cpp ExponentialLayer.cpp
        SubTensorLayer.cpp GAN.cpp Net.cpp GNet.cpp DNet.cpp ScaleLayer.cpp GPUAttr.cpp)

if (${Use_GPU})
   set(CDLF_CUDA_Src TensorCuda.cu DeviceKernels.cu )
else(${Use_GPU})
   set(CDLF_CUDA_Src "")
endif(${Use_GPU})

add_library(CDLF SHARED ${CDLF_CXX_Src} ${CDLF_CUDA_Src})
set_target_properties(CDLF PROPERTIES CUDA_SEPARABLE_COMPILATION ON )


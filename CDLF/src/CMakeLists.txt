
if (${Use_GPU})
   # set(CMAKE_CUDA_COMPILER /opt/apps/cuda/9.1.85.2/bin/nvcc)

   # this must correctly config, otherwise GPU can not work correctly
   set(CMAKE_CUDA_FLAGS -arch=sm_70)   #Nvidia Titan V  or GV100 in COE-GPU or UI-GPU
   #set(CMAKE_CUDA_FLAGS -arch=sm_61)    #Nvidia GeForce GTX 1080 Ti in COE-GPU or UI-GPU
   #set(CMAKE_CUDA_FLAGS -arch=sm_60)   #Nvidia Tesla P100 in UI-GPU
endif(${Use_GPU})

message("Config StatsLab Notes:")
message("0  Courtesy of Keith O'Hara (https://github.com/kthohr/stats) for StatsLab")
message("1  download code from: https://github.com/kthohr/stats into your intall directory")
message("2  in Clion File/Setting/Build/CMake/CMakeOptions config -DSTATS_INCLUDE_DIR=The/Full/Path/Of/Stats/Include/Dir ")
message("*************************************")

#set(STATS_INCLUDE_DIR /Users/hxie1/Projects/stats/include)
include_directories(${STATS_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(CDLF_CXX_Src  Layer.cpp FCLayer.cpp
        MatrixLayer.cpp LeftMatrixLayer.cpp RightMatrixLayer.cpp
        InputLayer.cpp LossLayer.cpp FeedForwardNet.cpp
        statisTool.cpp ReLU.cpp NormalizationLayer.cpp Tensor.hpp TensorBlas.cpp Tools.cpp
        ConvolutionBasicLayer.cpp  ConvolutionLayer.cpp TransposedConvolutionLayer.cpp
        LinearLayer.cpp SoftmaxLayer.cpp MaxPoolingLayer.cpp
        IdentityLayer.cpp MergerLayer.cpp BranchLayer.cpp  RescaleLayer.cpp
        SigmoidLayer.cpp  CrossEntropyLoss.cpp VectorizationLayer.cpp ExponentialLayer.cpp
        SubTensorLayer.cpp GAN.cpp Net.cpp GNet.cpp DNet.cpp GPUAttr.cpp CPUAttr.cpp FileTools.cpp
        LossConvexExample1.cpp LossConvexExample2.cpp LossNonConvexExample1.cpp LossNonConvexExample2.cpp
        PaddingLayer.cpp SquareLossLayer.cpp DiceLossLayer.cpp AssemblyLossLayer.cpp
        ReshapeLayer.cpp ConcatenateLayer.cpp
        )

if (${Use_GPU})
   set(CDLF_CUDA_Src TensorCuda.cu TensorKernels.cu LayerCuda.cu LayerKernels.cu
                     Cudnn.cpp CudnnBasicConvolution.cpp
                     CudnnConvolution.cpp CudnnTransposedConvolution.cpp
                     CudnnActivation.cpp  CudnnSoftmax.cpp)
else(${Use_GPU})
   set(CDLF_CUDA_Src "")
endif(${Use_GPU})


add_library(CDLF SHARED ${CDLF_CXX_Src} ${CDLF_CUDA_Src})
set_target_properties(CDLF PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_link_libraries(CDLF ${IPP_LIBRARY} ${MKL_LIBRARY} )




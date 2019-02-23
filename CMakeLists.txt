cmake_minimum_required (VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)

message("General Notes:")
message("0  you need to set " ${CMAKE_BINARY_DIR}/CDLF " into your path environmental variable. ")
message("1  you need to set the directory including CDLFLib into your environment variable.")
message("2  you need to install Intel MKL and IPP library and set their directory from  https://software.intel.com/en-us/parallel-studio-xe")
message("3  you need to download ITK source code and build its binary version from https://itk.org/ITK/resources/software.html")
message("4  you need to get stats library source code from https://github.com/kthohr/stats.git")
message("5  you need to get TIPL library source code from https://github.com/frankyeh/TIPL")
message("6  in Clion File/Setting/Build/CMake/CMakeOptions config, there should be something like:")
message("-DITK_DIR=/Users/hxie1/Projects/ITK_build
-DSTATS_INCLUDE_DIR=/Users/hxie1/Projects/stats/include
-DTIPL_Dir=/Users/hxie1/Projects/TIPL
-DINTEL_Dir=/home/hxie1/intel
-DUse_GPU=1
-DDev_Debug=0")
message("4  if you use CPU, instead of GPU, above -DUse_GPU=0")
message("*************************************")


if (${Use_GPU})
    project (CDLF_Framework LANGUAGES CXX CUDA)
else(${Use_GPU})
    project (CDLF_Framework LANGUAGES CXX)
endif(${Use_GPU})

# set(Use_GPU TRUE) # already configed in Clion File/Setting/Build/CMake/CMakeOptions config,

if (${Use_GPU})
    add_compile_definitions(Use_GPU)
    include_directories (${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(CUDA_TOOLKIT_INCLUDE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set(CMAKE_LIBRARY_PATH ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64)
    find_package(CUDA QUIET)
    message("Info: Program compile with Cuda GPU.")
else(${Use_GPU})
    message("Info: Program compile with CPU, without GPU.")
endif(${Use_GPU})

if (${Dev_Debug})
    add_compile_definitions(Dev_Debug)
    message("Info: Program will print the excutive time of each layer.")
else(${Dev_Debug})
    message("Info: Program Dev_Debug is off, which, if on, will print the excutive time of each layer.")
endif(${Dev_Debug})


message("*************************************")
set(CDLF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CDLF/include)
include_directories(${CDLF_INCLUDE_DIR})
link_directories(${CMAKE_BINARY_DIR}/CDLF/src)

# for Intel MKL library
set(MKLROOT ${INTEL_Dir}/mkl)
set(MKL_INCLUDE_DIR ${MKLROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
# use intel link advisor below to get link options
# https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

# for single dynamic library with intel ICC and intel64 architeture
set(MKL_LINK_FlAGS  "  -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl  ")


#for Intel IPP library
set(IPPROOT ${INTEL_Dir}/ipp)
set(IPP_INCLUDE_DIR ${IPPROOT}/include)
include_directories(${IPP_INCLUDE_DIR})
set(IPP_LINK_FlAGS  " -L${IPPROOT}/lib/intel64 -lippi -lipps -lippcore -lippvm")

set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${IPP_LINK_FlAGS} ${MKL_LINK_FlAGS}  ")


# for dynamic link MKL library
set(MKL_LIBRARY ${MKLROOT}/lib/intel64/libmkl_intel_lp64.so
                ${MKLROOT}/lib/intel64/libmkl_core.so
                ${MKLROOT}/lib/intel64/libmkl_sequential.so
                ${MKLROOT}/lib/intel64/libmkl_rt.so)

# for dynamic link IPP library
set(IPP_LIBRARY    ${IPPROOT}/lib/intel64/libippcore.so
                   ${IPPROOT}/lib/intel64/libippvm.so
                   ${IPPROOT}/lib/intel64/libipps.so
                   ${IPPROOT}/lib/intel64/libippi.so )



if (${Use_GPU})
    # for cudnn
    set(Cudnn_INCLUDE_DIR ${Cudnn_Dir}/include)
    include_directories(Cudnn_INCLUDE_DIR)
    set(Cudnn_LIBRARY ${Cudnn_Dir}/lib64/libcudnn.so)
else(${Use_GPU})
    message("We don't use ${Cudnn_Dir} as compiling CPU version.")
endif(${Use_GPU})



#set(CDLFLib libCDLF.dll)  # for windows dll
set(CDLFLib libCDLF.so pthread  ${IPP_LIBRARY} ${MKL_LIBRARY} ${Cudnn_LIBRARY} )    # for Linux

add_subdirectory(CDLF)
add_subdirectory(Examples)
add_subdirectory(Interface3rdParty)
add_subdirectory(PreprocessingTools)







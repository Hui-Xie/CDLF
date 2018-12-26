

message("Config TIPL Notes:")
message("0  Courtesy of Fang-Cheng Yeh(https://github.com/frankyeh/TIPL) for Template Image Processsing Library")
message("1  download code from:https://github.com/frankyeh/TIPL into your intall directory")
message("2  in Clion File/Setting/Build/CMake/CMakeOptions config -DTIPL_Dir=The/Full/Path/Of/TIPL/Dir ")
message("*************************************")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")

#set(TIPL_Dir /Users/hxie1/Projects/TIPL)

include_directories(${TIPL_Dir})


set (TIPLIODir ../../Interface3rdParty/TIPL)
include_directories(${TIPLIODir})

set(TestTIPL_SRCS testTIPL.cpp)
add_executable(TestTIPL ${TestTIPL_SRCS})
target_link_libraries(TestTIPL ${CDLFLib} ${CUDA_LIBRARIES} )
project (CDLF_Framework)

cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)



set(CDLF_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CDLF)
include_directories(${CDLF_SRC_DIR})
link_directories(${CMAKE_BINARY_DIR}/CDLF)

message("General Notes:")
message("0  you need to set " ${CMAKE_BINARY_DIR}/CDLF " into your path environmental variable. ")
message("1  In Windows, after setting envionmental varialbles, you need to reboot your system.")
message("2  you need to set the directory including CDLFLib into your environment variable.")
message("3  in Clion File/Setting/Build/CMake/CMakeOptions config, there should be something like:")
message("-DITK_DIR=/Users/hxie1/Projects/ITK_build
-DSTATS_INCLUDE_DIR=/Users/hxie1/Projects/stats/include
-DTIPL_Dir=/Users/hxie1/Projects/TIPL")
message("*************************************")

#set(CDLFLib libCDLF.dll)  # for windows dll
set(CDLFLib libCDLF.so)    # for Linux

add_subdirectory(CDLF)
add_subdirectory(Examples)
add_subdirectory(Interface3rdParty)





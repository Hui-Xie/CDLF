project (CDLF_Framework)

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "-std=c++14")


add_subdirectory(CDLF)
set(CDLF_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CDLF)
include_directories(${CDLF_SRC_DIR})
link_directories(${CMAKE_BINARY_DIR}/CDLF)

message("*************************************")
message("Notes: you need to set " ${CMAKE_BINARY_DIR}/CDLF " into your path environmental variable. ")
message("In Windows, after setting envionmental varialbles, you need to reboot your system.")
message("In HPC, in order to support C++14, you need to use: module load gcc/5.4.0")
message("*************************************")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CDLFLib libCDLF.dll)
# you need to set the directory including CDLFLib into your environment variable.

add_subdirectory(Examples)
add_subdirectory(Interface3rdParty)





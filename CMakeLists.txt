project (CDLF_Framework)

cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_subdirectory(CDLF)
set(CDLF_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CDLF)
include_directories(${CDLF_SRC_DIR})
link_directories(${CMAKE_BINARY_DIR}/CDLF)

message("*************************************")
message("Notes: you need to set " ${CMAKE_BINARY_DIR}/CDLF " into your path environmental variable. ")
message("In Windows, after setting envionmental varialbles, you need to reboot your system.")
message("*************************************")

set(CDLFLib libCDLF.dll)

# you need to set the directory including CDLFLib into your environment variable.

add_subdirectory(examples)




project (CDLF_Framework)

cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)


add_subdirectory(CDLF)
set(CDLF_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/CDLF)
include_directories(${CDLF_SRC_DIR})
link_directories(${CMAKE_BINARY_DIR}/CDLF)

x

#set(CDLFLib libCDLF.dll)  # for windows dll
set(CDLFLib libCDLF.so)    # for Linux
# you need to set the directory including CDLFLib into your environment variable.

add_subdirectory(Examples)
add_subdirectory(Interface3rdParty)






#set(ITK_DIR D:/ITK_build)
#Or you need to config ITK_DIR in your evironment setting in clion IDE.
#    -DITK_DIR=theFullPathOfITK-BuildDir


#cmake_policy(SET CMP0075 NEW)  #Include file check macros honor CMAKE_REQUIRED_LIBRARIES.
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}" -Wno-dev ")


find_package(ITK REQUIRED)
include(${ITK_USE_FILE})


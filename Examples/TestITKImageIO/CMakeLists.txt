
message("Config ITK Notes:")
message("0  Courtesy of Kitware Company (https://itk.org/itkindex.html) for ITK library")
message("1  download code from https://itk.org/ITK/resources/software.html into your intall directory")
message("2  decompress and compile ITK version")
message("3  in Clion File/Setting/Build/CMake/CMakeOptions config -DITK_DIR=the/Full/Path/Of/ITK-Build/Dir")
message("*************************************")


set(ITK_DIR /Users/hxie1/Projects/ITK_build)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (ITKImageIODir ../../Interface3rdParty/ITK)
include_directories(${ITKImageIODir})

set(TestITKImageIOSRCS TestITKImageIO.cpp)
add_executable(TestITKImageIO ${TestITKImageIOSRCS})
target_link_libraries(TestITKImageIO  ${ITK_LIBRARIES} ${CDLFLib} )


# make sure ITK and CDLF to use same compiler to compile
# Below is ITK4.13.1 compile config
#CMAKE_AR                         /opt/apps/binutils/2.27/bin/ar
#CMAKE_BUILD_TYPE                 Release
#CMAKE_COLOR_MAKEFILE             ON
#CMAKE_CXX_COMPILER               /opt/apps/gcc/5.4.0/bin/c++
#CMAKE_CXX_COMPILER_AR            /opt/apps/gcc/5.4.0/bin/gcc-ar
#CMAKE_CXX_COMPILER_RANLIB        /opt/apps/gcc/5.4.0/bin/gcc-ranlib
#CMAKE_CXX_FLAGS
#CMAKE_CXX_FLAGS_DEBUG            -g
#CMAKE_CXX_FLAGS_MINSIZEREL       -Os -DNDEBUG
#CMAKE_CXX_FLAGS_RELEASE          -O3 -DNDEBUG
#CMAKE_CXX_FLAGS_RELWITHDEBINFO   -O2 -g -DNDEBUG
#CMAKE_C_COMPILER                 /opt/apps/gcc/5.4.0/bin/gcc
#CMAKE_C_COMPILER_AR              /opt/apps/gcc/5.4.0/bin/gcc-ar
#CMAKE_C_COMPILER_RANLIB          /opt/apps/gcc/5.4.0/bin/gcc-ranlib




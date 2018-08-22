
#set(ITK_DIR D:/ITK_build)
#Or you need to config ITK_DIR in your evironment setting in Clion IDE.
#    -DITK_DIR=theFullPathOfITK-BuildDir

message("Config ITK Notes:")
message("0  Courtesy of Kitware Company (https://itk.org/itkindex.html) for ITK library")
message("1  download code from https://itk.org/ITK/resources/software.html into your intall directory")
message("2  decompress and compile ITK version")
message("3  in Clion File/Setting/Build/CMake/CMakeOptions config -DITK_DIR=the/Full/Path/Of/ITK-Build/Dir")
message("*************************************")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (ITKImageIODir ../../Interface3rdParty/ITK)
include_directories(${ITKImageIODir})

set(TestITKImageIOSRCS TestITKImageIO.cpp)
add_executable(TestITKImageIO ${TestITKImageIOSRCS})
target_link_libraries(TestITKImageIO ${CDLFLib} ${ITK_LIBRARIES})


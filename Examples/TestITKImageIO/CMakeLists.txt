
#set(ITK_DIR D:/ITK_build) # for windows
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (ITKImageIODir ../../Interface3rdParty/ITK)
include_directories(${ITKImageIODir})

set(TestITKImageIOSRCS TestITKImageIO.cpp)
add_executable(TestITKImageIO ${TestITKImageIOSRCS})
target_link_libraries(TestITKImageIO ${CDLFLib} ${ITK_LIBRARIES})


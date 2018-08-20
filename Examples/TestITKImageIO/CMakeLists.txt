
set(ITK_DIR D:/ITK_build) # for windows
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (ITKRWriter ../../Interface3rdParty/ITK)
include_directories(${ITKRWriter})

set(TestITKImageIOSRCS TestITKImageIO.cpp)
add_executable(TestITKImageIO ${TestITKImageIOSRCS})
target_link_libraries(TestITKImageIO ${CDLFLib} ${ITK_LIBRARIES})


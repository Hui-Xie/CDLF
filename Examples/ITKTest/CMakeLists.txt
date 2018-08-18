
set(ITK_DIR D:/ITK_build) # for windows
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set (ITKRWriter ../../Interfaces3ndParty/ITK)
include_directories(${ITKRWriter})

set(ITKTestSRCS ITKTest.cpp)
add_executable(ITKTest ${ITKTestSRCS})
target_link_libraries(ITKTest ${CDLFLib} ${ITK_LIBRARIES})
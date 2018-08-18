
set (ITKRWriter ../../Interfaces3ndParty/ITK)
include_directories(${ITKRWriter})

set(ITKTestSRCS ITKTest.cpp)
add_executable(ITKTest ${ITKTestSRCS})
target_link_libraries(ITKTest ${CDLFLib})
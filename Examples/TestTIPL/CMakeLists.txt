
set(TIPL_Dir /Users/hxie1/Projects/TIPL)
include_directories(${TIPL_Dir})


set (TIPLIODir ../../Interface3rdParty/TIPL)
include_directories(${TIPLIODir})

set(TestTIPL_SRCS testTIPL.cpp)
add_executable(TestTIPL ${TestTIPL_SRCS})
target_link_libraries(TestTIPL ${CDLFLib} )
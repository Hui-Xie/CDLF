
#set(ITK_DIR /Users/hxie1/Projects/ITK_build)

find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(TestITK4DImageIOSRCS TestITK4DImageIO.cpp)
    add_executable(TestITK4DImageIO ${TestITK4DImageIOSRCS})
    target_link_libraries(TestITK4DImageIO  ${ITK_LIBRARIES} ${CDLFLib} )
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)


find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(Test3DSegmentationSrc TestSegmentation3D.cpp Segmentation3DNet.cpp)
    add_executable(Test3DSegmentation ${Test3DSegmentationSrc})
    target_link_libraries(Test3DSegmentation  ${ITK_LIBRARIES} ${CDLFLib} )
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)

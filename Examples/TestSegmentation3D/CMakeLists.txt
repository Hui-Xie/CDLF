
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(TestSegmentation3DSrc TestSegmentation3D.cpp Segmentation3DNet.cpp SegmentGNet.cpp
                              SegmentDNet.cpp StubNetForD.cpp
                              DataManager.cpp)
    add_executable(TestSegmentation3D ${TestSegmentation3DSrc})
    target_link_libraries(TestSegmentation3D  ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES} )
else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)

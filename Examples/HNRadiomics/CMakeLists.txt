
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(TrainSegmentV_SRC TrainSegmentV.cpp HNSegVNet.cpp HNDataManager.cpp ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TrainSegmentV ${TrainSegmentV_SRC})
    target_link_libraries(TrainSegmentV ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})

    set(TestSegmentV_SRC TestSegmentV.cpp HNSegVNet.cpp HNDataManager.cpp  ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TestSegmentV ${TestSegmentV_SRC})
    target_link_libraries(TestSegmentV ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})

    set(TrainSurvival_SRC TrainSurvival.cpp HNSurvivalNet.cpp HNSegVNet.cpp HNDataManager.cpp  HNClinicalDataMgr.cpp ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TrainSurvival ${TrainSurvival_SRC})
    target_link_libraries(TrainSurvival ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})

else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)
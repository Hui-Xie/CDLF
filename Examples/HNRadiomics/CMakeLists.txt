
find_package(ITK QUIET)
if(ITK_FOUND)
    include(${ITK_USE_FILE})

    set (ITKImageIODir ../../Interface3rdParty/ITK)
    include_directories(${ITKImageIODir})

    set(TrainHNRadiomicsSRC TrainHNRadiomics.cpp HNRadiomicsNet.cpp HNDataManager.cpp ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TrainHNRadiomics ${TrainHNRadiomicsSRC})
    target_link_libraries(TrainHNRadiomics ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})

    set(TestHNRadiomicsSRC TestHNRadiomics.cpp HNRadiomicsNet.cpp HNDataManager.cpp  HNClinicalDataMgr.cpp ${ITKImageIODir}/ITKDataManager.cpp)
    add_executable(TestHNRadiomics ${TestHNRadiomicsSRC})
    target_link_libraries(TestHNRadiomics ${ITK_LIBRARIES} ${CDLFLib} ${CUDA_LIBRARIES})

else(ITK_FOUND )
    message("Infor: we can not find ITK.")
endif(ITK_FOUND)
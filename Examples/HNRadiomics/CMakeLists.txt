
set(TrainHNRadiomicsSRC TrainHNRadiomics.cpp HNRadiomicsNet.cpp)
add_executable(TrainHNRadiomics ${TrainHNRadiomicsSRC})
target_link_libraries(TrainHNRadiomics ${CDLFLib} ${CUDA_LIBRARIES})

set(qsubTestMnistSrc qsubTestMnist.cpp)
add_executable(qsubTestMnist ${qsubTestMnistSrc})

set(qsubTest3DSegGANSrc qsubTest3DSegGAN.cpp)
add_executable(qsubTest3DSegGAN ${qsubTest3DSegGANSrc})


set(qloginGPUSrc qloginGPU.cpp)
add_executable(qloginGPU ${qloginGPUSrc})


set(qloginCPUSrc qloginCPU.cpp)
add_executable(qloginCPU ${qloginCPUSrc})

set(qsubTestCudaSrc qsubTestCuda.cpp)
add_executable(qsubTestCuda ${qsubTestCudaSrc})

set(qsubHNSegVSrc qsubHNSegV.cpp)
add_executable(qsubHNSegV ${qsubHNSegVSrc})

set(qsubHNSurvival_Src qsubHNSurvival.cpp)
add_executable(qsubHNSurvival ${qsubHNSurvival_Src})

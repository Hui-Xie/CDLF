
set(qsubTestMnistSrc qsubTestMnist.cpp)
add_executable(qsubTestMnist ${qsubTestMnistSrc})

set(qsubTest3DSegGANSrc qsubTest3DSegGAN.cpp)
add_executable(qsubTest3DSegGAN ${qsubTest3DSegGANSrc})


set(qloginSessionSrc qloginSession.cpp)
add_executable(qloginSession ${qloginSessionSrc})

set(qsubTestCudaSrc qsubTestCuda.cpp)
add_executable(qsubTestCuda ${qsubTestCudaSrc})
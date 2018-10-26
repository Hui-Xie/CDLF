//
// Created by Hui Xie on 8/18/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITKImageIO.h"
#include <string>

void printUsage(char* argv0){
    cout<<"Test ITK 3D image:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathInputFileNane fullPathOutputFilename"<<endl;
}


int main(int argc, char *argv[]){
    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string inputFilename = argv[1];
    const string outputFilename = argv[2];

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    ITKImageIO<float, 3> itkImageIO;

    Tensor<float> *pImage = nullptr;
    itkImageIO.readFile(inputFilename, pImage);

    //change value of pImage,
    vector<long> tensorSize = pImage->getDims();
    vector<long> halfTensorSize = tensorSize / 2;
    for (long i = halfTensorSize[0] - 20; i < halfTensorSize[0] + 20; ++i)
        for (long j = halfTensorSize[1] - 20; j < halfTensorSize[1] + 20; ++j)
            for (long k = halfTensorSize[2] - 20; k < halfTensorSize[2] + 20; ++k) {
                pImage->e(i, j, k) = 0;  //dig a hole in the middle of brain.
            }

    itkImageIO.writeFileWithSameInputDim(pImage, {0, 0, 0}, outputFilename);

    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    cout << "============= End of TestITKImageIO =============" << endl;
}
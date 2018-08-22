//
// Created by Hui Xie on 8/18/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITKImageIO.h"
#include <string>

const string inputFilename = "/Users/hxie1/temp/BRATS_001.nii";
const string outputFilename = "//Users/hxie1/temp/BRATS_001_Output.nii";


int main(int argc, char *argv[]) {
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

    itkImageIO.writeFile(pImage, {0, 0, 0}, outputFilename);

    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    cout << "=============End of ITK Read Writer=============" << endl;
}
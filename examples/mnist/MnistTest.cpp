//
// Created by Sheen156 on 8/6/2018.
//

#include "MnistTools.h"

const string MnistDir= "E:\\CProject\\mnist";

int main (int argc, char *argv[])
{
    string trainImageFile = MnistDir + "\\train-images.idx3-ubyte";
    string trainLabelFile = MnistDir + "\\train-labels.idx1-ubyte";
    string testImageFile =  MnistDir +"\\t10k-images.idx3-ubyte";
    string testLabelFile = MnistDir + "\\t10k-labels.idx1-ubyte";



    Tensor<unsigned char>* pTensorMnistTestLabel = nullptr;

    readMNISTIdxFile(trainImageFile, pTensorMnistTestLabel);

    if (nullptr != pTensorMnistTestLabel){
        delete pTensorMnistTestLabel;
    }

    return 0;
}
//
// Created by Sheen156 on 8/6/2018.
//

#include "MnistTools.h"

const string MnistDir= "E:\\CProject\\mnist";

int main (int argc, char *argv[])
{
    string MnistTestLabelFile = MnistDir + "\\t10k-labels.idx1-ubyte";
    Tensor<unsigned char>* pTensorMnistTestLabel;

    readMNISTIdxFile(MnistTestLabelFile, pTensorMnistTestLabel);

    delete pTensorMnistTestLabel;
    return 0;
}
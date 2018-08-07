//
// Created by Sheen156 on 8/6/2018.
//

#ifndef CDLF_FRAMEWORK_MMISTTOOLS_H
#define CDLF_FRAMEWORK_MMISTTOOLS_H

#include "Tensor.h"

int readMNISTIdxFile(const string& fileName, Tensor<unsigned char>* pTensor);

#endif //CDLF_FRAMEWORK_MMISTTOOLS_H

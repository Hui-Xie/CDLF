//
// Created by Sheen156 on 8/8/2018.
//

#ifndef CDLF_FRAMEWORK_MNIST_H
#define CDLF_FRAMEWORK_MNIST_H

#include "Tensor.h"


class MNIST {
public:
    MNIST(const string& MnistDir);
    ~MNIST();

    Tensor<unsigned char> * m_pTrainImages;
    Tensor<unsigned char> * m_pTrainLabels;
    Tensor<unsigned char> * m_pTestImages;
    Tensor<unsigned char> * m_pTestLabels;

    void loadData();



private:
    string m_mnistDir;
    string m_trainImageFile;
    string m_trainLabelFile;
    string m_testImageFile;
    string m_testLabelFile;

    int readIdxFile(const string &fileName, Tensor<unsigned char> *pTensor);
    long hexChar4ToLong(char *buff);



};


#endif //CDLF_FRAMEWORK_MNIST_H

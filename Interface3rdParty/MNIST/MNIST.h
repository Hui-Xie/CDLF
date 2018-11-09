//
// Created by Hui Xie on 8/8/2018.
//

#ifndef CDLF_FRAMEWORK_MNIST_H
#define CDLF_FRAMEWORK_MNIST_H

#include "Tensor.h"


class MNIST {
public:
    MNIST(const string& mnistDir, bool onlyTestSet = false);
    ~MNIST();

    Tensor<unsigned char> * m_pTrainImages;
    Tensor<unsigned char> * m_pTrainLabels;
    Tensor<unsigned char> * m_pTestImages;
    Tensor<unsigned char> * m_pTestLabels;

    //as original MNIST dataset is too big, we tailor a part.
    Tensor<unsigned char> * m_pTrainImagesPart;
    Tensor<unsigned char> * m_pTrainLabelsPart;
    Tensor<unsigned char> * m_pTestImagesPart;
    Tensor<unsigned char> * m_pTestLabelsPart;

    vector<unsigned char> m_partDigits;

    void deleteWholeDataSet();
    void deletePartDataSet();

    void loadData();
    void tailorData();
    void displayImage(Tensor<unsigned char>* pImages, const long index);

private:
    string m_mnistDir;
    string m_trainImageFile;
    string m_trainLabelFile;
    string m_testImageFile;
    string m_testLabelFile;

    bool m_bOnlyTestSet;

    int readIdxFile(const string &fileName, Tensor<unsigned char>* &pTensor);
    long hexChar4ToLong(char *buff);
    bool isDigitInVector(const unsigned char digit);
    void extractPart(const Tensor<unsigned char> * pWholeImages,  const Tensor<unsigned char> * pWholeLabels,
                             Tensor<unsigned char> * pPartImages,  Tensor<unsigned char> *  pPartLabels);
};


#endif //CDLF_FRAMEWORK_MNIST_H

//
// Created by Hui Xie on 8/6/2018.
//

#ifndef CDLF_FRAMEWORK_MMISTTOOLS_H
#define CDLF_FRAMEWORK_MMISTTOOLS_H

#include "CDLF.h"
#include "MNIST.h"

class MnistConvNet: public Net{
public:
    MnistConvNet(MNIST* pMnistData);
    ~MnistConvNet();

    void setNetParameters();
    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    virtual void build();
    virtual void train();
    virtual float test();

    MNIST* m_pMnistData;


};


#endif //CDLF_FRAMEWORK_MMISTTOOLS_H

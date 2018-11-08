//
// Created by Hui Xie on 8/6/2018.
//

#ifndef CDLF_FRAMEWORK_MMISTTOOLS_H
#define CDLF_FRAMEWORK_MMISTTOOLS_H

#include "CDLF.h"
#include "MNIST.h"

class MnistConvNet: public FeedForwardNet{
public:
    MnistConvNet(const string& name, const string& saveDir, MNIST* pMnistData);
    ~MnistConvNet();

    Tensor<float> constructGroundTruth(Tensor<unsigned char> *pLabels, const long index);

    virtual void build();
    virtual void train();
    virtual float test();

    MNIST* m_pMnistData;

    void build4DConvolutionNet();
    void build2DConvolutionNet();


};


#endif //CDLF_FRAMEWORK_MMISTTOOLS_H

//
// Created by Hui Xie on 11/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#ifndef CDLF_FRAMEWORK_MEANSQUARELOSS_H
#define CDLF_FRAMEWORK_MEANSQUARELOSS_H

#include "LossLayer.h"

/* L= lambda*(0.5/N)*\sum (x_i- g_i)^2
 * where g_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       lambda is super parameter as weight for Mean square loss
 *       N is the number of elements.
 * */

class MeanSquareLossLayer  : public LossLayer {
public:
    MeanSquareLossLayer(const int id, const string& name,  Layer *prevLayer, float lambda = 1.0);
    ~MeanSquareLossLayer();
    virtual void  printGroundTruth();

    float m_lambda;

    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);
    float diceCoefficient(const float threshold);
    float getTPR(const float threshold); // TruePositiveRate = recall= sensitivity = TP/(TP+FN)

    template<typename ValueType> void getPredictTensor(Tensor<ValueType>& predictResult, const float threthold);

private:
    virtual float lossCompute();
    virtual void  gradientCompute();


};

template<typename ValueType>
void MeanSquareLossLayer::getPredictTensor(Tensor<ValueType>& predictResult, const float threshold) {
    const Tensor<float>* pPredict = m_prevLayer->m_pYTensor;
    const int N = pPredict->getLength();
    for (int i=0; i< N; ++i)
    {
        predictResult.e(i) = (pPredict->e(i) >= threshold)? 1:0;
    }

}


#endif //CDLF_FRAMEWORK_MEANSQUARELOSS_H

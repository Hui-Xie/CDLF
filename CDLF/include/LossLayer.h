//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_LOSSLAYER_H
#define CDLF_FRAME_LOSSLAYER_H

#include "Layer.h"
#include "Tensor.h"

//LossLayer is an abstract class.
//LossLayer has no learning parameters.

class LossLayer : public Layer {
public:
    LossLayer(const int id, const string& name, Layer *prevLayer);
    ~LossLayer();

    float getLoss();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void averageParaGradient(const int batchSize);
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);


    virtual  void initializeLRs(const float lr);
    virtual  void updateLRs(const float deltaLoss);
    virtual  void updateParameters(Optimizer* pOptimizer);

    virtual float lossCompute()=0;
    virtual void  gradientCompute()=0;
    virtual void  printGroundTruth();
    float m_loss;

    Tensor<float>* m_pGroundTruth;

    template<typename T> void setGroundTruth( const Tensor<T>& groundTruth);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();


    // only for loss after sigmoid
    float diceCoefficient(const float threshold);
    float getTPR(const float threshold); // TruePositiveRate = recall= sensitivity = TP/(TP+FN)
    template<typename ValueType> void getPredictTensor(Tensor<ValueType>& predictResult, const float threthold);


    //for softmax over feature channels
    float diceCoefficient();
    float getTPR(); // TruePositiveRate = recall= sensitivity = TP/(TP+FN)



};

template<typename T>
void LossLayer::setGroundTruth( const Tensor<T>& groundTruth){
    if (nullptr == m_pGroundTruth){
        m_pGroundTruth = new Tensor<float> (groundTruth.getDims());
    }
    const int N = m_pGroundTruth->getLength();
    for (int i=0; i<N; ++i){
        m_pGroundTruth->e(i) = (float) groundTruth.e(i);
    }
}

template<typename ValueType>
void LossLayer::getPredictTensor(Tensor<ValueType>& predictResult, const float threshold) {
    const Tensor<float>* pPredict = m_prevLayer->m_pYTensor;
    const int N = pPredict->getLength();
    for (int i=0; i< N; ++i)
    {
        predictResult.e(i) = (pPredict->e(i) >= threshold)? 1:0;
    }

}


#endif //CDLF_FRAME_LOSSLAYER_H

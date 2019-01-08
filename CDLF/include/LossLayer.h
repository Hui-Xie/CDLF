//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#ifndef RL_NONCONVEX_LOSSLAYER_H
#define RL_NONCONVEX_LOSSLAYER_H

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
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);

    virtual float lossCompute()=0;
    virtual void  gradientCompute()=0;
    virtual void printGroundTruth()=0;
    float m_loss;

    Tensor<float>* m_pGroundTruth;

    template<class T> void setGroundTruth( const Tensor<T>& groundTruth);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct(const int layerIndex);

};

template<class T>
void LossLayer::setGroundTruth( const Tensor<T>& groundTruth){
    if (nullptr == m_pGroundTruth){
        m_pGroundTruth = new Tensor<float> (groundTruth.getDims());
    }
    const int N = m_pGroundTruth->getLength();
    for (int i=0; i<N; ++i){
        m_pGroundTruth->e(i) = (float) groundTruth.e(i);
    }
}


#endif //RL_NONCONVEX_LOSSLAYER_H

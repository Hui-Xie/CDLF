//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#ifndef CDLF_FRAME_INPUTLAYER_H
#define CDLF_FRAME_INPUTLAYER_H

#include "Layer.h"



class InputLayer : public Layer{
public:
    InputLayer(const int id, const string& name,const vector<int>& tensorSize);
    ~InputLayer();

    virtual  void initialize(const string& initialMethod);
    virtual  void zeroParaGradient();
    virtual  void forward();
    virtual  void backward(bool computeW, bool computeX = true);
    virtual  void updateParameters(const float lr, const string& method, const int batchSize=1);
    template<typename T> void setInputTensor(const Tensor<T>& inputTensor);
    virtual  int getNumParameters();

    virtual  void save(const string& netDir);
    virtual  void load(const string& netDir);
    virtual  void saveStructLine(FILE* pFile);
    virtual  void printStruct();
};

template<typename T>
void InputLayer::setInputTensor(const Tensor<T> &inputTensor) {
    if (m_tensorSize == inputTensor.getDims()) {
        const int N = m_pYTensor->getLength();
        for (int i=0; i<N; ++i){
            m_pYTensor->e(i) = (float) inputTensor.e(i);
        }
    } else {
        cout << "Error: setInputTensor(const Tensor<T>& inputTensor) has different tensorSize." << endl;
    }
}


#endif //CDLF_FRAME_INPUTLAYER_H

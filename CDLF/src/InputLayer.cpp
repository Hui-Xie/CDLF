//
// Created by Hui Xie on 6/6/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <InputLayer.h>

#include "InputLayer.h"
#include "statisTool.h"
#include "Tools.h"

InputLayer::InputLayer(const int id, const string& name, const vector<long>& tensorSize): Layer(id, name, tensorSize){
    m_type = "InputLayer";

}

InputLayer::~InputLayer(){

}

void InputLayer::initialize(const string& initialMethod){
    //null
}

void InputLayer::zeroParaGradient(){
   //null
}

void InputLayer::forward(){
    //null
}
void InputLayer::backward(bool computeW){
    //null
}

void InputLayer::updateParameters(const float lr, const string& method, const int batchSize){
    //null
}

void InputLayer::setInputTensor(const Tensor<float>& inputTensor){
    if (m_tensorSize == inputTensor.getDims()){
        *m_pYTensor = inputTensor;
    }
    else{
        cout<<"Error: setInputTensor(const Tensor<float>& inputTensor) has different tensorSize."<<endl;
    }
}

void InputLayer::setInputTensor(const Tensor<unsigned char> &inputTensor) {
    if (m_tensorSize == inputTensor.getDims()) {
        long N = m_pYTensor->getLength();
#ifdef Use_GPU
        cudaElementCopy(inputTensor.getData(), m_pYTensor->getData(), N);
#else
        for (long i=0; i<N; ++i){
            m_pYTensor->e(i) = (float) inputTensor.e(i);
        }
#endif
    } else {
        cout << "Error: setInputTensor(const Tensor<unsigned char>& inputTensor) has different tensorSize." << endl;
    }

}

long InputLayer::getNumParameters() {
    return 0;
}

void InputLayer::save(const string &netDir) {
   //null
}

void InputLayer::load(const string &netDir) {
  //null
}

void InputLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), 0, vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

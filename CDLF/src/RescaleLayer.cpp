//
// Created by Hui Xie on 12/17/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.

#include <RescaleLayer.h>

/*
 *  Y = k*(X- Xmin)/(Xmax -Xmin);
 *  dL/dX = dL/dY * k/(Xmax -Xmin)
 *
 * */

RescaleLayer::RescaleLayer(const int id, const string& name,Layer* prevLayer, float k): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "RescaleLayer";
    m_k = k;
    if ( 0 == m_k) {
        cout <<"Error: k in RescaleLayer should not equal to zero."<<endl;
    }
    addPreviousLayer(prevLayer);
}

RescaleLayer::~RescaleLayer(){

}

void RescaleLayer::forward() {
    Tensor<float> &Y = *m_pYTensor;
    Tensor<float> &X = *m_prevLayer->m_pYTensor;
    float min = 0;
    float max = 0;
    X.getMinMax(min, max);
    float kdiff = max - min;
    if (0 == kdiff) {
        return; // all Y.e(i)  =0,which already compute in the forward propagation.
    }
    else {
        kdiff = m_k / kdiff;
        const int N = Y.getLength();
        for (int i = 0; i < N; ++i) {
            Y.e(i) = (X.e(i) - min) * kdiff;
        }
    }
}

void RescaleLayer::backward(bool computeW, bool computeX) {
    if (computeX) {
        Tensor<float> &dY = *m_pdYTensor;
        Tensor<float> &dX = *m_prevLayer->m_pdYTensor;
        Tensor<float> &X = *m_prevLayer->m_pYTensor;

        float min = 0;
        float max = 0;
        X.getMinMax(min, max);
        float kdiff = max - min;
        if (0 == kdiff) {
            return;// all dX.e(i) should equal 0
        }
        else {
            kdiff = m_k / kdiff;
            const int N = dY.getLength();
            for (int i = 0; i < N; ++i) {
                if (max != X.e(i) && min != X.e(i)) {
                    dX.e(i) += dY.e(i) * kdiff;
                }// when X.e(i) ==max or X.e(i) == min, dX.e(i) should equal 0.
            }
        }
    }
}
void RescaleLayer::initialize(const string& initialMethod){
    //null
}

void RescaleLayer::zeroParaGradient(){
    //null
}

void RescaleLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  RescaleLayer::getNumParameters(){
    return 0;
}

void RescaleLayer::save(const string &netDir) {
    //null
}

void RescaleLayer::load(const string &netDir) {
    //null
}

void RescaleLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, m_k, "{}");
}

void RescaleLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s,  PrevLayer=%s, k=%f, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),   m_prevLayer->m_name.c_str(), m_k, vector2Str(m_tensorSize).c_str());
}

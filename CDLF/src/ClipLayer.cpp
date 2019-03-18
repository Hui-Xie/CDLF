//
// Created by Hui Xie on 03/15/2019.
// Copyright (c) 2019 Hui Xie. All rights reserved.

#include <ClipLayer.h>

ClipLayer::ClipLayer(const int id, const string& name, Layer* prevLayer, const int min, const int max): Layer(id,name, prevLayer->m_tensorSize) {
    m_type = "ClipLayer";
    m_min = min;
    m_max = max;
    if (m_max <= m_min){
        cout<<"Error: in ClipLayer max should be greater than min. "<<endl;
        std::exit(-1);
    }
    addPreviousLayer(prevLayer);
}

ClipLayer::~ClipLayer(){

}

//gives x for min≤x≤max, min for x<min and max for x>max.
void ClipLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    Tensor<float>& X = *m_prevLayer->m_pYTensor;
    const int N = Y.getLength();
    for (int i=0; i< N; ++i){
       if (X.e(i) > m_max ) Y.e(i) = m_max;
       else if (X.e(i) < m_min ) Y.e(i) = m_min;
       else Y.e(i) = X.e(i);
    }
}

void ClipLayer::backward(bool computeW, bool computeX) {
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;
        Tensor<float>& dX = *m_prevLayer->m_pdYTensor;
        Tensor<float>& X = *m_prevLayer->m_pYTensor;
        const int N = dY.getLength();
        for(int i=0; i< N; ++i){
            if (X.e(i) >= m_min && X.e(i) <= m_max) dX.e(i) += dY.e(i);
            // all dX.e(i) = 0 in zeroDYTensor() method in each iteration.
        }
    }
}
void ClipLayer::initialize(const string& initialMethod){
    //null
}

void ClipLayer::zeroParaGradient(){
    //null
}

void ClipLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  ClipLayer::getNumParameters(){
    return 0;
}

void ClipLayer::save(const string &netDir) {
    //null
}

void ClipLayer::load(const string &netDir) {
    //null
}

void ClipLayer::saveStructLine(FILE *pFile) {
    string minMaxStr=to_string(m_min)+"_"+to_string(m_max);
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %f, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0.0, minMaxStr.c_str());
}

void ClipLayer::printStruct() {
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, min=%d, max=%d, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  m_prevLayer->m_name.c_str(), m_min, m_max, vector2Str(m_tensorSize).c_str());
}

void ClipLayer::initializeLRs(const float lr) {

}

void ClipLayer::updateLRs(const float deltaLoss, const int batchSize) {

}

void ClipLayer::updateParameters(const string &method, const int batchSize) {

}

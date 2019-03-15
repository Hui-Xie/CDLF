//
// Created by Hui Xie on 12/21/18.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include <ConcatenateLayer.h>

ConcatenateLayer::ConcatenateLayer(const int id, const string& name, const vector<Layer*>& layersVec, const vector<int>& tensorSize)
        : Layer(id,name, tensorSize){
    m_type = "ConcatenateLayer";
    if (layersVec.size() <= 1){
        cout<<"Error: the previous Layers in ConcatenateLayer should be greater than 1"<<endl;
        return;
    }
    m_layersVec = layersVec;
    addPreviousLayer(m_layersVec[0]);

    const int N = m_layersVec.size();
    int totalLen = 0;
    for (int i =0; i<N ;++i){
        m_layerOffsetVec.push_back(totalLen);
        const int layerLength = m_layersVec[i]->m_pYTensor->getLength();
        m_layerLengthVec.push_back(layerLength);
        totalLen += layerLength;
    }

    if (totalLen != length(m_tensorSize)){
        cout<<"Error: the output TensorSize does not euqal the sum of previous Layers, at layID = "<<id<<endl;
    }
}

ConcatenateLayer::~ConcatenateLayer(){
   //null
}

void ConcatenateLayer::forward(){
    Tensor<float>& Y = *m_pYTensor;
    const int N = m_layerLengthVec.size();
    for(int i = 0; i<N; ++i){
        Tensor<float>& X = *m_layersVec[i]->m_pYTensor;
        Y.copyDataFrom(X.getData(), m_layerLengthVec[i], m_layerOffsetVec[i]);
    }
}
void ConcatenateLayer::backward(bool computeW, bool computeX){
    if (computeX){
        Tensor<float>& dY = *m_pdYTensor;

        // this code is simpler, but a little inefficient
        //const int N = dY.getLength();
        //for (int i=0; i<N; ++i){
        //    dX(i) += dY.e(i);
        //}

        //=================
        const int N = m_layersVec.size();
        for (int i =0; i< N; ++i){
            const int layerLength = m_layerLengthVec[i];
            for (int j=0; j< layerLength; ++j){
                m_layersVec[i]->m_pdYTensor->e(j) += dY.e(m_layerOffsetVec[i]+j);
            }
        }
    }
}
void ConcatenateLayer::initialize(const string& initialMethod){
    //null
}

void ConcatenateLayer::zeroParaGradient(){
    //null
}

void ConcatenateLayer::updateParameters(const float lr, const string& method, const int batchSize) {
    //null
}

int  ConcatenateLayer::getNumParameters(){
    return 0;
}

void ConcatenateLayer::save(const string &netDir) {
//null
}

void ConcatenateLayer::load(const string &netDir) {
//null
}

void ConcatenateLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    string prevLayersStr="";
    const int N = m_layersVec.size();
    for (int i=0; i<N; ++i){
        prevLayersStr += to_string(m_layersVec[i]->m_id) + ((N-1 == i)?"":"_");
    }
    fprintf(pFile, "%d, %s, %s, %s, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), prevLayersStr.c_str(),
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}

void ConcatenateLayer::printStruct() {
    string prevLayersStr="";
    const int N = m_layersVec.size();
    for (int i=0; i<N; ++i){
        prevLayersStr += to_string(m_layersVec[i]->m_id) + ((N-1 == i)?"":"_");
    }
    printf("id=%d, Name=%s, Type=%s, PrevLayer=%s, OutputSize=%s; \n",
           m_id, m_name.c_str(),m_type.c_str(),  prevLayersStr.c_str(), vector2Str(m_tensorSize).c_str());
}

float& ConcatenateLayer::dX(const int index) const {
    const int N = m_layerOffsetVec.size();
    int i = N-1;
    while (index < m_layerOffsetVec[i]){
        --i;
    }
    return m_layersVec[i]->m_pdYTensor->e(index- m_layerOffsetVec[i]);
}

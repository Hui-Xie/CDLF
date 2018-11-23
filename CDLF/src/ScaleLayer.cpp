//
// Created by Hui Xie on 9/28/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <ScaleLayer.h>

#include "ScaleLayer.h"


ScaleLayer::ScaleLayer(const int id, const string& name, Layer* prevLayer,  const float k): Layer(id,name, prevLayer->m_tensorSize){
    m_type = "ScaleLayer";
    addPreviousLayer(prevLayer);
    m_k = k;
    m_dk = 0;
}

ScaleLayer::~ScaleLayer() {
   //null
}


void ScaleLayer::initialize(const string& initialMethod){
    //null
}

void ScaleLayer::zeroParaGradient(){
    m_dk = 0;
}

//Y = k*X
void ScaleLayer::forward(){
    *m_pYTensor = *m_prevLayer->m_pYTensor * m_k;
}

/* y_i = k*x_i, where k is a learning scalar.
 * dL/dx_i = dL/dy_i *k
 * dL/dk = (dL/dy)' * x, where y and x are 1D vector form,prime symbol means transpose.
 *
 * */
void ScaleLayer::backward(bool computeW){
    if (computeW) {
        m_dk += m_pdYTensor->dotProduct(*m_prevLayer->m_pYTensor);
    }
    *(m_prevLayer->m_pdYTensor) += *m_pdYTensor * m_k;
}

void ScaleLayer::updateParameters(const float lr, const string& method, const int batchSize){
    if ("sgd" == method){
        m_k -=  m_dk*(lr/batchSize);
    }
}

long ScaleLayer::getNumParameters(){
    return 1;
}

void ScaleLayer::save(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    createDir(layerDir);

    filename= layerDir + "/K.csv";
    pFile = fopen (filename.c_str(),"w");
    if (nullptr == pFile){
        printf("Error: can not open  %s  file  in writing.\n", filename.c_str());
        return;
    }
    fprintf(pFile, "%f ", m_k);
    fprintf(pFile,"\r\n");
    fclose (pFile);
}

void ScaleLayer::load(const string &netDir) {
    FILE * pFile = nullptr;
    string filename = "";

    string layerDir = netDir + "/" + to_string(m_id);
    if (!dirExist(layerDir)){
        initialize("Xavier");
        return;
    }
    else{
        filename= layerDir + "/K.csv";
        pFile = fopen (filename.c_str(),"r");
        if (nullptr == pFile){
            printf("Error: can not open  %s  file for reading.\n", filename.c_str());
            return;
        }
        fscanf(pFile, "%f ", &m_k);
        fclose (pFile);
    }
}

void ScaleLayer::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, NumFilter, FilterStride(k), StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", 0, 0, "{}");
}

void ScaleLayer::printStruct(const int layerIndex) {
    printf("Layer%03d, Name=%s: (%s, id=%d): PrevLayer=%s, OutputSize=%s; \n",
           layerIndex, m_name.c_str(),m_type.c_str(), m_id,  m_prevLayer->m_name.c_str(), vector2Str(m_tensorSize).c_str());
}

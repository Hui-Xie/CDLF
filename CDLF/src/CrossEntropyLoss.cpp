//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved  .

#include <CrossEntropyLoss.h>

#include "CrossEntropyLoss.h"

CrossEntropyLoss::CrossEntropyLoss(const int id, const string& name, Layer *prevLayer ): LossLayer(id,name,prevLayer){
    m_type = "CrossEntropyLoss";
}

CrossEntropyLoss::~CrossEntropyLoss(){

}

/* L= -(1/N)*\sum (p_i * log(x_i) + (1-p_i) *log(1 -x_i))
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       */

float CrossEntropyLoss::lossCompute(){
    //X.e \in [0,1]
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    const int N = X.getLength();
    m_loss  = 0;
    for (int i=0; i< N; ++i){
        float  x = X.e(i);
        float  g = m_pGroundTruth->e(i);
        if (x == g){
            continue;
        }
        else{
            if (x < 0.1){
                x = 0.1;
            }
            if (x > 0.9){
                x = 0.9;
            }
            m_loss += -g*log(x)-(1-g)*log(1-x);
        }
    }
    m_loss /=N;
    return m_loss;
}

// L= -(1/N)*\sum (p_i * log(x_i) + (1-p_i) *log(1 -x_i))
// dL/dx_i = (- p_i/x_i + (1-p_i)/(1-x_i))/N
// this formula implies it is better for p_i is one-hot vector;
// and we need to check X.e(i) ==0  and X.e(i) ==1 case.

void CrossEntropyLoss::gradientCompute() {
    //symbol deduced formula to compute gradient to prevLayer->m_pdYTensor
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    const int N = dX.getLength();
    for (int i = 0; i < N; ++i) {
        float x = X.e(i);
        float g = m_pGroundTruth->e(i);
        if (x == g){
            continue;
        }
        else{
            if (x < 0.1){
                x = 0.1;
            }
            if (x > 0.9){
                x = 0.9;
            }
            dX[i] += (-g/x +(1-g)/(1-x))/N;
        }
    }
}

void  CrossEntropyLoss::printGroundTruth() {
    m_pGroundTruth->print();
}

bool CrossEntropyLoss::predictSuccessInColVec(){
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    Tensor<float> &Y = *m_pGroundTruth;
    if (X.maxPosition() == Y.maxPosition()) {
        return true;
    }
    else{
        return false;
    }
}

float CrossEntropyLoss::diceCoefficient(){
    Tensor<float> &predict = *(m_prevLayer->m_pYTensor);
    Tensor<float> &GT = *m_pGroundTruth;
    Tensor<unsigned char> predictMaxPosTensor = predict.getMaxPositionSubTensor();
    Tensor<unsigned char> GTMaxPosTensor = GT.getMaxPositionSubTensor();
    const int N = predictMaxPosTensor.getLength();
    int nPredict = 0;
    int nGT = 0;
    int nInteresection = 0;
    for (int i=0; i< N; ++i)
    {
        nPredict += (0 !=predictMaxPosTensor(i))? 1: 0;
        nGT      += (0 !=GTMaxPosTensor(i))? 1: 0;
        nInteresection += (predictMaxPosTensor(i) == GTMaxPosTensor(i) && 0 != predictMaxPosTensor(i)) ? 1:0;
    }
    return nInteresection*2.0/(nPredict+nGT);
}

float CrossEntropyLoss::getTPR() {
    Tensor<float> &predict = *(m_prevLayer->m_pYTensor);
    Tensor<float> &GT = *m_pGroundTruth;
    Tensor<unsigned char> predictMaxPosTensor = predict.getMaxPositionSubTensor();
    Tensor<unsigned char> GTMaxPosTensor = GT.getMaxPositionSubTensor();
    const int N = predictMaxPosTensor.getLength();
    int nTP = 0; // True Positive
    int nP = 0;// nP = nTP + nFP
    for (int i=0; i< N; ++i)
    {
        if (GTMaxPosTensor.e(i) >= 0)
        {
            ++nP;
            if (predictMaxPosTensor.e(i) ==  GTMaxPosTensor.e(i) ){
                ++nTP;
            }
        }
    }
    return nTP*1.0/nP;
}

int CrossEntropyLoss::getNumParameters(){
    return 0;
}

void CrossEntropyLoss::save(const string &netDir) {
   //null
}

void CrossEntropyLoss::load(const string &netDir) {
   //null
}

void CrossEntropyLoss::saveStructLine(FILE *pFile) {
    //const string tableHead= "ID, Type, Name, PreviousLayerIDs, OutputTensorSize, FilterSize, Stride, NumFilter, k/lambda, StartPosition, \r\n"
    fprintf(pFile, "%d, %s, %s, %d, %s, %s, %s, %d, %d, %s, \r\n", m_id, m_type.c_str(), m_name.c_str(), m_prevLayer->m_id,
            vector2Str(m_tensorSize).c_str(), "{}", "{}", 0, 0, "{}");
}



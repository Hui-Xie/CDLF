//
// Created by Hui Xie on 8/8/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved  .

#include "CrossEntropyLoss.h"

CrossEntropyLoss::CrossEntropyLoss(const int id, const string& name, Layer *prevLayer ): LossLayer(id,name,prevLayer){
    m_type = "CrossEntropyLoss";
}

CrossEntropyLoss::~CrossEntropyLoss(){

}

/* L= -\sum p_i * log(x_i)
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       */

float CrossEntropyLoss::lossCompute(){
    //X.e \in [0,1]
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    m_loss = m_pGroundTruth->hadamard(X.ln()).sum()*(-1);
    return m_loss;
}

// L= -\sum p_i * log(x_i)
// dL/dx_i = - p_i/x_i
// this formula implies it is better for p_i is one-hot vector;
// and we need to check X.e(i) ==0 case.

void CrossEntropyLoss::gradientCompute() {
    //symbol deduced formula to compute gradient to prevLayer->m_pdYTensor
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    long N = dX.getLength();
    for (long i = 0; i < N; ++i) {
        if (0 != X.e(i)){
            dX[i] -= m_pGroundTruth->e(i)/X.e(i);
        }
        else{
            dX[i] -= m_pGroundTruth->e(i)/(1e-5);
        }

    }
}

void  CrossEntropyLoss::printGroundTruth() {
    cout << "For this specific Loss function, Ground Truth is: ";
    m_pGroundTruth->printElements(false);
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
    Tensor<int> predictMaxPosTensor = predict.getMaxPositionSubTensor();
    Tensor<int> GTMaxPosTensor = GT.getMaxPositionSubTensor();
    const long N = predictMaxPosTensor.getLength();
    if (N != GTMaxPosTensor.getLength()){
        cout <<"Error: predicting Tensor has a different dimension with groundtruth"<<endl;
        return -1;
    }
    long nSuccess = 0;
    for (long i=0; i< N; ++i)
    {
        if (predictMaxPosTensor(i) == GTMaxPosTensor(i)){
            ++nSuccess;
        }
    }
    return nSuccess*1.0/N;

}

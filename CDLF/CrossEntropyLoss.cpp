//
// Created by Sheen156 on 8/8/2018.
//

#include "CrossEntropyLoss.h"

CrossEntropyLoss::CrossEntropyLoss(const int id, const string& name): LossLayer(id,name){
    m_type = "CrossEntropyLoss";
}
CrossEntropyLoss::~CrossEntropyLoss(){

}

/* L= -\sum p_i * log(x_i)
 * where p_i is the groundtruth distribution
 *       x_i is the output of previous layer, e.g. softmax;
 *       */

float CrossEntropyLoss::lossCompute(Tensor<float>* pGroundTruth){
    //use m_prevLayerPointer->m_pYTensor,
    m_loss = 0;
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    m_loss = pGroundTruth->hadamard(X.ln()).sum()*(-1);
    return m_loss;
}

// L= -\sum p_i * log(x_i)
// dL/dx_i = - p_i/x_i
void CrossEntropyLoss::gradientCompute() {
    //symbol deduced formula to compute gradient to prevLayer->m_pdYTensor
    Tensor<float> &prevY = *(m_prevLayer->m_pYTensor);
    Tensor<float> &prevdY = *(m_prevLayer->m_pdYTensor);
    long N = prevY.getLength();
    for (long i = 0; i < N; ++i) {
        prevdY[i] += 2 * (prevY[i] - i);
    }
}

void  CrossEntropyLoss::printGroundTruth() {
    cout << "For this specific Loss function, Ground Truth is: ";
    long N = m_prevLayer->m_pYTensor->getLength();
    cout << "( ";
    for (long i = 0; i < N; ++i) {
        if (i != N - 1) cout << i << ", ";
        else cout << i;
    }
    cout << " )" << endl;
}

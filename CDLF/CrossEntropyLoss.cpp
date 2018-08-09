//
// Created by Hui Xie on 8/8/2018.
//

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
    //use m_prevLayerPointer->m_pYTensor,
    Tensor<float> & X = *(m_prevLayer->m_pYTensor);
    m_loss = m_pGroundTruth->hadamard(X.ln()).sum()*(-1);
    return m_loss;
}

// L= -\sum p_i * log(x_i)
// dL/dx_i = - p_i/x_i
void CrossEntropyLoss::gradientCompute() {
    //symbol deduced formula to compute gradient to prevLayer->m_pdYTensor
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
    long N = dX.getLength();
    for (long i = 0; i < N; ++i) {
        dX[i] -= m_pGroundTruth->e(i)/X.e(i);
    }
}

void  CrossEntropyLoss::printGroundTruth() {
    cout << "For this specific Loss function, Ground Truth is: ";
    m_pGroundTruth->printElements(false);
}

bool CrossEntropyLoss::predictSuccess(){
    Tensor<float> &X = *(m_prevLayer->m_pYTensor);
    Tensor<float> &Y = *m_pGroundTruth;
    if (X.maxPosition() == Y.maxPosition()) {
        return true;
    }
    else{
        return false;
    }
}

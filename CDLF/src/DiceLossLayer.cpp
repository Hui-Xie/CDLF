//
// Created by Hui Xie on 02/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <DiceLossLayer.h>

DiceLossLayer::DiceLossLayer(const int id, const string &name, Layer *prevLayer) : LossLayer(id, name, prevLayer) {
    m_type = "DiceLossLayer";

    if ("SigmoidLayer" != prevLayer->m_type  &&
       ("SoftmaxLayer" != prevLayer->m_type  || 2 != prevLayer->m_tensorSize[0]))
    {
        cout<<"Error: DiceLossLayer should follow with SigmoidLayer, or SoftmaxLayer with feature dimension 2."<<endl;
        std:exit(EXIT_FAILURE);
    }
}

DiceLossLayer::~DiceLossLayer() {
    //null;
}

float DiceLossLayer::lossCompute() {
    if ("SigmoidLayer" == m_prevLayer->m_type){
        Tensor<float> & X = *(m_prevLayer->m_pYTensor);
        // here nom is L1 norm, when x>0, g>0, L1norm = sum

        const float xDotg_norm = X.hadamard(*m_pGroundTruth).sum();
        const float xPlusg_norm = X.sum()+ m_pGroundTruth->sum();
        if (0 == xPlusg_norm) {
            return 1;
        }
        m_loss = 1 -2.0* xDotg_norm/ xPlusg_norm;
    }
    else {// for SoftmaxLayer as previous Layer
        Tensor<float>* pX1 = nullptr;
        m_prevLayer->m_pYTensor->extractLowerDTensor(1, pX1);
        Tensor<float>* pG1 = nullptr;
        m_pGroundTruth->extractLowerDTensor(1, pG1);

        // here nom is L1 norm, when x>0, g>0, L1norm = sum
        const float xDotg_norm = pX1->hadamard(*pG1).sum();
        const float xPlusg_norm = pX1->sum()+ pG1->sum();
        if (0 == xPlusg_norm) {
            return 1;
        }
        m_loss = 1 -2.0* xDotg_norm/ xPlusg_norm;

        delete pX1;
        delete pG1;
    }

    return m_loss;
}

void DiceLossLayer::gradientCompute() {
    if ("SigmoidLayer" == m_prevLayer->m_type) {
        Tensor<float> &X = *(m_prevLayer->m_pYTensor);
        Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);
        Tensor<float> &G = *m_pGroundTruth;

        // here norm is L1 norm, when x>0, g>0, L1norm = sum
        const int N = X.getLength();
        const float xDotg_norm = X.hadamard(*m_pGroundTruth).sum();
        const float xPlusg_norm = X.sum() + m_pGroundTruth->sum();
        const float xPlusg_norm2 = 2.0 / (xPlusg_norm * xPlusg_norm);
        for (int i = 0; i < N; ++i) {
            dX[i] += (xDotg_norm - G.e(i) * xPlusg_norm) * xPlusg_norm2;
        }

    } else {// for SoftmaxLayer as previous Layer
        Tensor<float> *pX1 = nullptr;
        m_prevLayer->m_pYTensor->extractLowerDTensor(1, pX1);
        Tensor<float> *pG1 = nullptr;
        m_pGroundTruth->extractLowerDTensor(1, pG1);

        Tensor<float> &dX = *(m_prevLayer->m_pdYTensor);

        // here norm is L1 norm, when x>0, g>0, L1norm = sum
        const int N = pX1->getLength();
        const float xDotg_norm = pX1->hadamard(*pG1).sum();
        const float xPlusg_norm = pX1->sum() + pG1->sum();
        if (xPlusg_norm > 0 ){
            const float xPlusg_norm2 = 2.0 / (xPlusg_norm * xPlusg_norm);
            for (int i = 0; i < N; ++i) {
                dX[i+N] += (xDotg_norm - pG1->e(i) * xPlusg_norm) * xPlusg_norm2;
                dX[i] -= dX[i+N];
            }
        }
        else{
            cout<<"Error: xPlusg_norm <0 in DiceLoss Layer."<<endl;
        }

        delete pX1;
        delete pG1;
    }
}





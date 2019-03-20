//
// Created by Hui Xie on 8/6/2018.
//

#include "MnistConvNet.h"

MnistConvNet::MnistConvNet(const string& saveDir, MNIST* pMnistData): FeedForwardNet(saveDir){
  m_pMnistData = pMnistData;
}

MnistConvNet::~MnistConvNet(){

}

void MnistConvNet::build(){
   //null
}


// train one epoch
void MnistConvNet::train(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    const int N =m_pMnistData->m_pTrainLabels->getLength();
    const int batchSize = getBatchSize();
    const float learningRate = getLearningRate();
    const int numBatch = (N + batchSize -1) / batchSize;
    int nIter = 0;
    int nBatch = 0;
    //random reshuffle data samples
    vector<int> randSeq = generateRandomSequence(N);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < N; ++i) {
            inputLayer->setInputTensor(m_pMnistData->m_pTrainImages->slice(randSeq[nIter]));
            lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabels, randSeq[nIter]));
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        averageParaGradient(i);
        optimize("SGD");
        ++nBatch;
    }
}


float MnistConvNet::test(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    int n = 0;
    int nSuccess = 0;
    const int Ntest = m_pMnistData->m_pTestLabels->getLength();
    while (n < Ntest) {
        inputLayer->setInputTensor(m_pMnistData->m_pTestImages->slice(n));
        lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabels, n));
        forwardPropagate();
        if (lossLayer->predictSuccessInColVec()) ++nSuccess;
        ++n;
    }
    cout<<"Info: nSuccess = "<<nSuccess<<" in "<<Ntest<<" test samples."<<endl;
    return  nSuccess * 1.0 / Ntest;
}

//construct a 2*1 one-hot vector
Tensor<float> MnistConvNet::constructGroundTruth(Tensor<unsigned char> *pLabels, const int index) {
    Tensor<float> tensor({10, 1});
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1;
    return tensor;
}



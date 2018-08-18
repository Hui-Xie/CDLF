//
// Created by Hui Xie on 8/6/2018.
//

#include "MnistConvNet.h"

MnistConvNet::MnistConvNet(MNIST* pMnistData){
  m_pMnistData = pMnistData;
}

MnistConvNet::~MnistConvNet(){

}

void MnistConvNet::setNetParameters() {
    setLearningRate(0.001);
    setLossTolerance(0.02);
    setBatchSize(100);
    initialize();

}

void MnistConvNet::build(){
    // it is a good design if all numFilter is odd;
    int layerID = 1;
    InputLayer *inputLayer = new InputLayer(layerID++, "InputLayer", {28, 28}); //output size: 28*28
    addLayer(inputLayer);

    ConvolutionLayer *conv1 = new ConvolutionLayer(layerID++, "Conv1", {3, 3},getFinalLayer(), 5, 1); //output size: 5*26*26
    addLayer(conv1);
    ReLU *reLU1 = new ReLU(layerID++, "ReLU1", getFinalLayer());
    addLayer(reLU1);
    NormalizationLayer *norm1 = new NormalizationLayer(layerID++, "Norm1", getFinalLayer());
    addLayer(norm1);

    ConvolutionLayer *conv2 = new ConvolutionLayer(layerID++, "Conv2", {5,3,3}, getFinalLayer(), 5, 1);//output size: 5*24*24
    addLayer(conv2);
    ReLU *reLU2 = new ReLU(layerID++, "ReLU2", getFinalLayer());
    addLayer(reLU2);
    NormalizationLayer *norm2 = new NormalizationLayer(layerID++, "Norm2", getFinalLayer());
    addLayer(norm2);

    ConvolutionLayer *conv3 = new ConvolutionLayer(layerID++, "Conv3", {3,3, 3}, getFinalLayer(), 7, 1);//output size: 7*3*22*22
    addLayer(conv3);
    ReLU *reLU3 = new ReLU(layerID++, "ReLU3", getFinalLayer());
    addLayer(reLU3);
    NormalizationLayer *norm3 = new NormalizationLayer(layerID++, "Norm3", getFinalLayer());
    addLayer(norm3);

    ConvolutionLayer *conv4 = new ConvolutionLayer(layerID++, "Conv4", {7,3,3,3}, getFinalLayer(), 1, 1);//output size: 20*20
    addLayer(conv4);
    ReLU *reLU4 = new ReLU(layerID++, "ReLU4", getFinalLayer());
    addLayer(reLU4);
    NormalizationLayer *norm4 = new NormalizationLayer(layerID++, "Norm4", getFinalLayer());
    addLayer(norm4);

    VectorizationLayer *vecLayer1 = new VectorizationLayer(layerID++, "Vector1", getFinalLayer()); //output size: 400*1
    addLayer(vecLayer1);
    FCLayer *fcLayer1 = new FCLayer(layerID++, "FC1", 100, getFinalLayer()); //output size: 100*1
    addLayer(fcLayer1);
    ReLU *reLU5 = new ReLU(layerID++, "ReLU5", getFinalLayer()); //output size: 100*1
    addLayer(reLU5);
    NormalizationLayer *norm5 = new NormalizationLayer(layerID++, "Norm5", getFinalLayer());
    addLayer(norm5);

    FCLayer *fcLayer2 = new FCLayer(layerID++, "FC2", 10, getFinalLayer()); //output size: 10*1
    addLayer(fcLayer2);
    ReLU *reLU6 = new ReLU(layerID++, "ReLU6", getFinalLayer());
    addLayer(reLU6);
    NormalizationLayer *norm6 = new NormalizationLayer(layerID++, "Norm6", getFinalLayer());
    addLayer(norm6);

    //For 2 category case
    FCLayer *fcLayer3 = new FCLayer(layerID++, "FC3", 2, getFinalLayer()); //output size: 2*1
    addLayer(fcLayer3);

    SoftMaxLayer *softmaxLayer = new SoftMaxLayer(layerID++, "Softmax1",getFinalLayer()); //output size: 2*1
    addLayer(softmaxLayer);
    CrossEntropyLoss *crossEntropyLoss = new CrossEntropyLoss(layerID++, "CrossEntropy",
                                                              getFinalLayer()); // output size: 1
    addLayer(crossEntropyLoss);
}

void MnistConvNet::train(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();

    long maxIteration =m_pMnistData->m_pTrainLabelsPart->getLength();
    long NTrain = maxIteration;
    int batchSize = getBatchSize();
    float learningRate = getLearningRate();
    long numBatch = maxIteration / batchSize;
    if (0 != maxIteration % batchSize) {
        numBatch += 1;
    }

    long nIter = 0;
    long nBatch = 0;
    //random reshuffle data samples
    vector<long> randSeq = generateRandomSequence(NTrain);
    while (nBatch < numBatch) {
        zeroParaGradient();
        int i = 0;
        for (i = 0; i < batchSize && nIter < maxIteration; ++i) {
            inputLayer->setInputTensor(m_pMnistData->m_pTrainImagesPart->slice(randSeq[nIter]));
            lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTrainLabelsPart, randSeq[nIter]));
            forwardPropagate();
            backwardPropagate();
            ++nIter;
        }
        sgd(learningRate, i);
        ++nBatch;
    }
}

float MnistConvNet::test(){
    InputLayer *inputLayer = getInputLayer();
    CrossEntropyLoss *lossLayer = (CrossEntropyLoss *) getFinalLayer();
    long n = 0;
    long nSuccess = 0;
    const long Ntest = m_pMnistData->m_pTestLabelsPart->getLength();
    while (n < Ntest) {
        inputLayer->setInputTensor(m_pMnistData->m_pTestImagesPart->slice(n));
        lossLayer->setGroundTruth(constructGroundTruth(m_pMnistData->m_pTestLabelsPart, n));
        forwardPropagate();
        if (lossLayer->predictSuccess()) ++nSuccess;
        ++n;
    }
    cout<<"Info: nSuccess = "<<nSuccess<<" in "<<Ntest<<" test samples."<<endl;
    return  nSuccess * 1.0 / Ntest;
}

//construct a 2*1 one-hot vector
Tensor<float> MnistConvNet::constructGroundTruth(Tensor<unsigned char> *pLabels, const long index) {
    Tensor<float> tensor({2, 1});// for 2 categories case
    tensor.zeroInitialize();
    tensor.e(pLabels->e(index)) = 1; //for {0,1} case
    return tensor;
}



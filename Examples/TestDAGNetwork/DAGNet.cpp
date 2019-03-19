//
// Created by Hui Xie on 8/29/18.
//

#include "DAGNet.h"
#include "LossConvexExample1.h"
/*
 *
 *
 * Input-->FC1-->Branch-->|----->FC2-------> FC3----->|-->Merger-->FC4-->Loss
 *                        |                           |
 *                        |-->Conv1-->Conv2-->Conv3-->|
 *
 * */


DAGNet::DAGNet(const string& saveDir): FeedForwardNet(saveDir){

}

DAGNet::~DAGNet(){

}

void DAGNet::buildSimple(){
    int id=1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {10,1});
    addLayer(inputLayer);

    //add parallel convolution path
    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", inputLayer, {3,1}, {1,1} ); //output: {8,1}
    addLayer(conv1);

    //add Loss Layer
    LossConvexExample1* loss = new LossConvexExample1(id++, "LossLayer", conv1);
    addLayer(loss);

}


void DAGNet::build(){
    // add head path
    int id=1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {10,1});
    addLayer(inputLayer);
    FCLayer* fc1 = new FCLayer(id++, "FC1", getFinalLayer(), 20);
    addLayer(fc1);
    ReLU* reLU1 = new ReLU(id++, "ReLU1", getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(reLU1);
    NormalizationLayer* normalLayer1 = new NormalizationLayer(id++, "NormLayer1",getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(normalLayer1);

    BranchLayer* branch = new BranchLayer(id++, "Branch", getFinalLayer()); //output: {20,1}
    addLayer(branch);

    //add parallel convolution path
    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", branch, {5,1}, {1,1} ); //output: {16,1}
    addLayer(conv1);
    ReLU* reLU2 = new ReLU(id++, "ReLU2", getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(reLU2);
    NormalizationLayer* normalLayer2 = new NormalizationLayer(id++, "NormLayer2",getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(normalLayer2);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", getFinalLayer(),{3,1}, {1,1}); //output: {14,1}
    addLayer(conv2);
    ReLU* reLU3 = new ReLU(id++, "ReLU3", getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(reLU3);
    NormalizationLayer* normalLayer3 = new NormalizationLayer(id++, "NormLayer3",getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(normalLayer3);

    ConvolutionLayer* conv3 = new ConvolutionLayer(id++, "Conv3", getFinalLayer(), {3,1}, {1,1}); //output: {12,1}
    addLayer(conv3);
    ReLU* reLU4 = new ReLU(id++, "ReLU4", getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(reLU4);
    NormalizationLayer* normalLayer4 = new NormalizationLayer(id++, "NormLayer4",getFinalLayer(), getFinalLayer()->m_tensorSize);
    addLayer(normalLayer4);

    //add parallel FC path
    FCLayer* fc2 = new FCLayer(id++, "FC2", branch, 15); //output {15,1}
    addLayer(fc2);
    ReLU* reLU5 = new ReLU(id++, "ReLU5", fc2,fc2->m_tensorSize);
    addLayer(reLU5);
    NormalizationLayer* normalLayer5 = new NormalizationLayer(id++, "NormLayer5",reLU5, reLU5->m_tensorSize);
    addLayer(normalLayer5);

    FCLayer* fc3 = new FCLayer(id++, "FC3", normalLayer5, 12); //output {12,1}
    addLayer(fc3);
    ReLU* reLU6 = new ReLU(id++, "ReLU6", fc3,fc3->m_tensorSize);
    addLayer(reLU6);
    NormalizationLayer* normalLayer6 = new NormalizationLayer(id++, "NormLayer6",reLU6, reLU6->m_tensorSize);
    addLayer(normalLayer6);

    //add MergerLayer
    MergerLayer* merger = new MergerLayer(id++, "MergerLayer", {12,1});
    addLayer(merger);
    merger->addPreviousLayer(normalLayer4);
    merger->addPreviousLayer(normalLayer6);

    //add tail path
    FCLayer* fc4 = new FCLayer(id++, "FC4", merger, 10);
    addLayer(fc4);

    //add Loss Layer
    LossConvexExample1* loss = new LossConvexExample1(id++, "LossLayer", fc4);
    addLayer(loss);
}

void DAGNet::train(){
    int nIter = 0;
    InputLayer* inputLayer = (InputLayer*)getInputLayer();
    LossLayer* lossLayer = (LossLayer* ) getFinalLayer();
    int maxIteration = 100;
    int batchSize = getBatchSize();
    float lr = getLearningRate();
    int numBatch =  (maxIteration + batchSize -1) / batchSize;

    int nBatch = 0;
    while(nBatch < numBatch)
    {
        if (getJudgeLoss() && lossLayer->getLoss()< getLossTolerance()){
            break;
        }
        if (isinf(lossLayer->getLoss())) break;

        zeroParaGradient();
        int i=0;
        for(i=0; i< batchSize && nIter < maxIteration; ++i){
            Tensor<float> inputTensor(inputLayer->m_pYTensor->getDims());
            generateGaussian(&inputTensor, 0,1);
            inputLayer->setInputTensor(inputTensor);
            forwardPropagate();
            backwardPropagate(true);
            ++nIter;
        }
        sgd(lr);
        printIteration(lossLayer, nIter);
        ++nBatch;
    }
    lossLayer->printGroundTruth();

}

float DAGNet::test(){
   //null;
}
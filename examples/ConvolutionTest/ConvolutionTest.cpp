//
// Created by Hui Xie on 8/13/2018.
//
#include "CDLF.h"

/*  inputLayer: 5*5
 *  ConvLayer: 1 filter of size 3*3;
 *  VectorizationLayer: 9*1
 *  LossConvexExmaple1: 1
 *
 *  * */

#include "LossConvexExample1.h"

int main (int argc, char *argv[])
{
    Net net;

    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {5,5});
    net.addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", {3,3}, net.getFinalLayer());
    net.addLayer(conv1);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", net.getFinalLayer());
    net.addLayer(vec1);

    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", net.getFinalLayer());
    net.addLayer(loss);

    // config network parameters;
    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setMaxIteration(100);
    net.setBatchSize(1);

    //  run network
    net.initialize();

    Tensor<float> inputTensor({5,5});
    inputTensor.uniformIntialize(1);
    long i=0;
    while (i< 100){
        inputLayer->setInputTensor(inputTensor);
        net.forwardPropagate();
        net.backwardPropagate();
        net.printIteration(loss, i);
        ++i;
    }



    cout<< "End of Convolution Test."<<endl;
    return 0;
}
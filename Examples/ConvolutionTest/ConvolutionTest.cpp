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
    cout<<"Notes:"<<endl;
    cout<<"This program test that 2 simple convolutional layers can approximate a convex function, and converge."<<endl;
    cout<<"This program support real 3D convolution."<<endl;

    Net net;
    // build network
    int id =1;
    InputLayer* inputLayer = new InputLayer(id++, "InputLayer", {7,7});
    net.addLayer(inputLayer);

    ConvolutionLayer* conv1 = new ConvolutionLayer(id++, "Conv1", {3,3}, net.getFinalLayer(),3); //output 3*5*5
    net.addLayer(conv1);

    NormalizationLayer* norm1 = new NormalizationLayer(id++, "Norm1",net.getFinalLayer());
    net.addLayer(norm1);

    ConvolutionLayer* conv2 = new ConvolutionLayer(id++, "Conv2", {3,3,3}, net.getFinalLayer()); //output 3*3
    net.addLayer(conv2);

    VectorizationLayer* vec1 = new VectorizationLayer(id++, "Vec1", net.getFinalLayer());
    net.addLayer(vec1);

    LossConvexExample1* loss = new LossConvexExample1(id++, "Loss", net.getFinalLayer());
    net.addLayer(loss);

    // config network parameters;
    net.setLearningRate(0.001);
    net.setLossTolerance(0.02);
    net.setBatchSize(1);
    net.printArchitecture();

    //  run network
    net.initialize();

    Tensor<float> inputTensor({7,7});
    generateGaussian(&inputTensor,0,1);
    inputLayer->setInputTensor(inputTensor);

    cout<<endl<<"Start to Train"<<endl;
    long i=0;
    while (i< 200){
        net.zeroParaGradient();
        net.forwardPropagate();
        net.printIteration(loss, i);
        net.backwardPropagate();
        net.sgd(net.getLearningRate(), 1);
        ++i;
    }
    loss->printGroundTruth();

    cout<< "=========== End of ConvolutionLayer Test ============"<<endl;
    return 0;
}
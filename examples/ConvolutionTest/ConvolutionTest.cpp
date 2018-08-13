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
    // run network




    cout<< "End of Convolution Test."<<endl;
    return 0;
}
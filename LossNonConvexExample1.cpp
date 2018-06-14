//
// Created by Sheen156 on 6/13/2018.
//

#include "LossNonConvexExample1.h"
LossNonConvexExample1::LossNonConvexExample1(Layer* preLayer): LossLayer(preLayer){

}
LossNonConvexExample1::~LossNonConvexExample1(){

}

//f(x,y) = 3ysin(x)+5xcos(y)+0.5xy+x^2-y^2
//Loss = f -0;
//dL/dx = dL/df* df/dx = 3ycos(x)+ 5cos(y) + 0.5y +2x
//dL/dy = dL/df* df/dx = 3sin(x)- 5xsin(y)+ 0,5x-2y
float LossNonConvexExample1::lossCompute(){

}

void  LossNonConvexExample1::gradientCompute(){

}

void  LossNonConvexExample1::printGroundTruth(){

}
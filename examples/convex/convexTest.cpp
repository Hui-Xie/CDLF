//
// Created by Hui Xie on 6/5/2018.
//

#include "convexTest.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "Net.h"
#include "LossConvexExample1.h"
#include "LossConvexExample2.h"

using namespace std;

void printUsage(){
    cout<<"A Fully-Connected Network compute loss function using statistic gradient descent."<<endl;
    cout<<"Usage: cmd layerWidthVector"<<endl
        <<"For example: cmd 5,7,8,10,5"<<endl
        <<"Where layerWidthVector use comma as separator, and it does not include ReLU layers and Normlization Layers."<<endl;

}

int convertCommaStrToVector(const string commaStr, std::vector<long>& widthVector){
    string widthStr = commaStr;
    int NChar = widthStr.length();
    for (int i=0; i< NChar; ++i){
        if (','==widthStr.at(i)) widthStr.at(i) = ' ';
    }
    std::stringstream iss(widthStr);
    long number;
    while ( iss >> number ){
        widthVector.push_back( number );
    }

    if (widthVector.size() > 0) return 0;
    else return -1;

}

int main (int argc, char *argv[])
{
    if (2 != argc){
        cout<<"Input parameter error. Exit"<<endl;
        printUsage();
        return -1;
    }
    string stringLayersWidth = string(argv[1]);
    vector<long> layerWidthVector;
    int result = convertCommaStrToVector(stringLayersWidth, layerWidthVector);
    if (0 != result){
        cout<<"Layer width string has error. Exit."<<endl;
        return -1;
    }
    Net net;

    //convex example 1: f= \sum (x_i-i)^2
    LossConvexExample1* lossLayer = new LossConvexExample1(1000, "ConvexLossLayer");

    //convex example 2: f= = \sum exp(x_i -i)
    //LossConvexExample2* lossLayer = new LossConvexExample2(1003, "ConvexLossLayer");

    net.buildFullConnectedNet(layerWidthVector, lossLayer);
    net.setLearningRate(0.01);
    net.setLossTolerance(0.02);
    net.setMaxIteration(1000);
    net.setBatchSize(20);
    net.initialize();

    net.train();
    std::cout<<"====================End of This Program==================="<<std::endl;
    return 0;

}

//
// Created by Hui Xie on 8/29/18.
//

#include "DAGNet.h"

int main (int argc, char *argv[])
{
    DAGNet net("DAGNet");
    net.build();
    net.printArchitecture();

    net.setLearningRate(0.01);
    net.setLossTolerance(0.02);
    net.setBatchSize(20);
    net.initialize();

    net.train();
    net.test();
    std::cout<<"====================End of This Program==================="<<std::endl;
    return 0;
}
//
// Created by Hui Xie on 6/5/2018.
//

#include "UnitFilterNet.h"
#include "LossConvexExample1.h"
#include <sstream>

using namespace std;

void printUsage(){
   //null
}

int main (int argc, char *argv[])
{
    UnitFilterNet net;

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

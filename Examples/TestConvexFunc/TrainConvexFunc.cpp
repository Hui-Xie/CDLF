//
// Created by Hui Xie on 6/5/2018.
//

#include "convexTest.h"
#include "ConvexNet.h"
#include "LossConvexExample1.h"
#include "LossConvexExample2.h"
#include <sstream>

using namespace std;

void printUsage(char* argv0){
    cout<<"A Fully-Connected Network compute loss function using statistic gradient descent."<<endl;
    cout<<"Usage: "<<endl
        <<argv0<<" <netDirectory> "<<endl
        <<"Where:"<<endl
        <<"netDirectory: the storing directory of net parameters. If not empty, program will load network from this directory instead build network using layerWidthVector"<<endl;

}

int convertCommaStrToVector(const string commaStr, std::vector<int>& widthVector){
    string widthStr = commaStr;
    int NChar = widthStr.length();
    for (int i=0; i< NChar; ++i){
        if (','==widthStr.at(i)) widthStr.at(i) = ' ';
    }
    std::stringstream iss(widthStr, ios_base::in);
    int number;
    while ( iss >> number ){
        widthVector.push_back( number );
    }

    if (widthVector.size() > 0) return 0;
    else return -1;

}

int main (int argc, char *argv[])
{
    printCurrentLocalTime();
    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    if (2 != argc){
        cout<<"Input parameter error. Exit"<<endl;
        printUsage(argv[0]);
        return -1;
    }
    string netDir= string(argv[1]);

    ConvexNet net(netDir);

    AdamOptimizer adamOptimizer(0.001,0.9,0.999);
    net.setOptimizer(&adamOptimizer);

    net.load();
    //net.setLearningRate(0.01);
    net.printArchitecture();

    net.train();
    net.test();
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    printCurrentLocalTime();

    net.save();

    return 0;

}

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
        <<argv0<<"<netDirectory>  <layerWidthVector>  "<<endl
        <<"Where:"<<endl
        <<"netDirectory: the storing directory of net parameters. If not empty, program will load network from this directory instead build network using layerWidthVector"<<endl
        <<"layerWidthVector: e.g.  5,7,8,10,5   uses comma as separator, and it does not include ReLU layers and Normlization Layers."<<endl;


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

    if (3 != argc){
        cout<<"Input parameter error. Exit"<<endl;
        printUsage(argv[0]);
        return -1;
    }
    string netDir= string(argv[1]);
    string stringLayersWidth = string(argv[2]);
    vector<int> layerWidthVector;
    int result = convertCommaStrToVector(stringLayersWidth, layerWidthVector);
    if (0 != result){
        cout<<"Layer width string has error. Exit."<<endl;
        return -1;
    }

    ConvexNet net("ConvexNet", netDir, layerWidthVector);

    if (isEmptyDir(net.getDir())){
        net.build();
        net.initialize();
        net.setLearningRate(0.01);
        net.setLossTolerance(0.02);
        net.setBatchSize(20);
    }
    else{
        net.load();
    }
    net.setLearningRate(1);
    net.printArchitecture();

    net.train();
    net.test();
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    printCurrentLocalTime();

    net.save();

    return 0;

}

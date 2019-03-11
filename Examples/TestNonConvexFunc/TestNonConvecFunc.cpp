//
// Created by Hui Xie on 6/5/2018.
//

#include "TestNonConvecFunc.h"
#include "NonconvexNet.h"
#include "LossNonConvexExample1.h"
#include "LossNonConvexExample2.h"
#include <sstream>

using namespace std;

void printUsage() {
    cout << "A Fully-Connected Network compute loss function using statistic gradient descent." << endl;
    cout << "Usage: cmd layerWidthVector" << endl
         << "For example: cmd 5,7,8,10,5" << endl
         << "Where layerWidthVector use comma as separator, and it does not include ReLU layers and Normlization Layers."
         << endl;

}

int convertCommaStrToVector(const string commaStr, std::vector<int> &widthVector) {
    string widthStr = commaStr;
    int NChar = widthStr.length();
    for (int i = 0; i < NChar; ++i) {
        if (',' == widthStr.at(i)) widthStr.at(i) = ' ';
    }
    std::stringstream iss(widthStr, ios_base::in);
    int number;
    while (iss >> number) {
        widthVector.push_back(number);
    }

    if (widthVector.size() > 0) return 0;
    else return -1;

}

int main(int argc, char *argv[]) {
    if (2 != argc) {
        cout << "Input parameter error. Exit" << endl;
        printUsage();
        return -1;
    }
    string stringLayersWidth = string(argv[1]);
    vector<int> layerWidthVector;
    int result = convertCommaStrToVector(stringLayersWidth, layerWidthVector);
    if (0 != result) {
        cout << "Layer width string has error. Exit." << endl;
        return -1;
    }

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout << "Info: program use CPU, instead of GPU." << endl;
#endif

    NonconvexNet net("./NonConvextNet", layerWidthVector);
    if (isEmptyDir(net.getDir())) {
        net.build();
        net.initialize();
        net.setJudgeLoss(false); //for nonconvex case
        net.setLearningRate(0.01);
        net.setLossTolerance(0.02);
        net.setBatchSize(30);
    } else {
        net.load();
    }


    net.train();
    net.test();
    net.save();
    cout << "=========== End of Test:  " << net.getName() << " ============" << endl;
    return 0;

}

// Created by Hui Xie on 01/12/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "HNSegVNet.h"

void printUsage(char* argv0){
    cout<<"Test Head&Neck Suqamous Cell Carcinoma Segmentation V Model:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfCTImageFile>  [groundTruthFile]"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/data/HeadNeckSCC/ExtractData/CT_Images/HNSCC-01-0017_CT.nrrd "
                 " /home/hxie1/data/HeadNeckSCC/ExtractData/GTV_Images/HNSCC-01-0017_GTV.nrrd"<<endl;
    cout<<argv0<<" /Users/hxie1/temp_netParameters /Users/hxie1/data/HeadNeckSCC/ExtractData/CT_Images/HNSCC-01-0017_CT.nrrd "
                 " /Users/hxie1/data/HeadNeckSCC/ExtractData/GTV_Images/HNSCC-01-0017_GTV.nrrd"<<endl;
    cout<<"Notes:"<<endl;
    cout<<"1  GroundTruth file is optional; if it is given, program will compute its dice coeffficient"<<endl;
}


int main(int argc, char *argv[]) {
    printCurrentLocalTime();
    if (4 != argc && 3 != argc) {
        cout << "Error: input parameter error." << endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string imageFile = argv[2];
    string labelFile = "";
    if (4 == argc) {
        labelFile = argv[3];
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

#ifdef Use_GPU
    HNSegVNet net("HNSCC_convV", netDir);
#else
    HNRadiomicsNet net("HNSCC_matrix", netDir);
#endif

    if (!isEmptyDir(net.getDir())) {
        net.load();
    } else {
        cout << "Error: program can not load net." << endl;
        return -2;
    }
    //net.defineAssemblyLoss();
    net.printArchitecture();
    net.detectSoftmaxBeforeLoss();

    HNDataManager dataMgr("");
    net.m_pDataMgr = &dataMgr;

    if (isContainSubstr(net.getName(),"clip")){
        net.m_pDataMgr->generateLabelCenterMap();
    }

    vector<int> center = vector<int>();
    if (!labelFile.empty()){
        center = net.m_pDataMgr->getLabelCenter(labelFile, false, 0);
    }

    net.test(imageFile, labelFile, center);

    cout << "=========== End of Predict:  " << net.getName() << " ============" << endl;
    return 0;

}
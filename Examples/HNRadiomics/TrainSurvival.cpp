//
// Created by Hui Xie on 02/16/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "HNSurvivalNet.h"

void printUsage(char* argv0){
    cout<<"Train Head&Neck Squamous Cell Carcinoma Survival"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<netDir> <fullPathOfRadiomicsDataDir>  <learningRate> <clinicalFilename>"<<endl;
    cout<<"For examples: "<<endl;
    cout<<argv0<<" /home/hxie1/temp_netParameters /home/hxie1/data/HeadNeckSCC/ExtractData  0.05 /home/hxie1/temp_netParameters/HNSCC_ROI_Survival/HNSCC_Clinical_survival.csv"<<endl;
    cout<<argv0<<" /Users/hxie1/temp_netParameters  /Users/hxie1/data/HeadNeckSCC/ExtractData 0.05 /Users/hxie1/temp_netParameters/HNSCC_ROI_Survival/HNSCC_Clinical_survival.csv"<<endl;
}


int main(int argc, char *argv[]){
    printCurrentLocalTime();
    if (5 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string netDir = argv[1];
    const string dataDir = argv[2];
    const float learningRate = stof(argv[3]);
    const string clinicalFile = argv[4];

    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

#ifdef Use_GPU
    HNSurvivalNet net("HNSCC_ROI_Survival", netDir);

#else
    HNSurvivalNet net("HNSCC_ROI_Survival", netDir);
#endif

    cout<<"=========================================="<<endl;
    cout<<"Info: this is "<<net.getName() <<" net."<<endl;
    cout<<"=========================================="<<endl;

    if (!isEmptyDir(net.getDir())) {
        net.load();
    }
    else{
        cout<<"Error: program can not load net."<<endl;
        return -2;
    }
    net.printArchitecture();
    net.setLearningRate(learningRate);
    net.setUnlearningLayerID(300);

    //for one sample training
    //net.setOneSampleTrain(true);

    HNDataManager dataMgr(dataDir);
    net.m_pDataMgr = &dataMgr;

    HNClinicalDataMgr clinicalDataMgr(clinicalFile);
    net.m_pClinicalDataMgr =&clinicalDataMgr;


    if (isContainSubstr(net.getName(),"ROI")){
        net.m_pDataMgr->generateLabelCenterMap();
    }

    int epoch= 15000;
    for (int i=0; i<epoch; ++i){
        cout <<"Epoch "<<i<<": "<<endl;
        net.train();
        if (0 == (i+1)%23 ){
            net.save();
        }

        net.test(0 == (i+1)%10 ? true : false);
    }
    cout<< "=========== End of Test:  "<<net.getName() <<" ============"<<endl;
    return 0;
}
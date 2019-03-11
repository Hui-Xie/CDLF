//
// Created by Hui Xie on 01/22/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
using namespace std;

string cmdPath = " /Users/hxie1/temp_release/Examples/HNRadiomics/TrainSegmentV ";
//string cmdPara = " /Users/hxie1/temp_netParameters/HNSCC_convV  /Users/hxie1/data/HeadNeckSCC/ExtractData 0.01 "; // for Conv-V model
//string cmdPara = " /Users/hxie1/temp_netParameters/HNSCC_matrix_V  /Users/hxie1/data/HeadNeckSCC/ExtractData 1 "; // for Matrix-V model
//string cmdPara = " /Users/hxie1/temp_netParameters/HSCC_convV  /Users/hxie1/data/HeadNeckSCC/ExtractData 0.005 "; // for ROI1-V model
string cmdPara = " /Users/hxie1/temp_netParameters/CATonsil  /Users/hxie1/data/HeadNeckSCC/CATonsil 0.005 "; // for CATonsil-V model


void printUsage(char* argv0){
    cout<<"Train HNSCC Radiomics in HPC"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <jobName> <Queue> <numSlots>  <0/1>"<<endl;
    cout<<"jobName: it will specify the output file in the ~/temp_qsub directory."<<endl;
    cout<<"GPUQueue: COE-GPU, UI-DEVELOP, UI-GPU, INFORMATICS-GPU, INFORMATICS-HM-GPU, all.q;  Choose only one." <<endl;
    cout<<"CPUQueue: COE, UI-DEVELOP, UI-HM, UI-MPI, INFORMATICS, all.q;  Choose only one." <<endl;
    cout<<"numSlots: number of slots"<<endl;
    cout<<"<0/1> indicates Not Use GPU(0), or Use GPU(1)"<<endl;
}


int main(int argc, char *argv[]) {
    // notes: all command paramaeters have a space at front and at tail
    if (5 != argc){
        printUsage(argv[0]);
        return -1;
    }
    string jobName = argv[1];
    string queue = argv[2];
    int numSlots = atoi(argv[3]);
    bool useGPU = bool(atoi(argv[4]));

    string gpuResouce= "gpu_titanv=true" ; // "gpu_1080ti=true" or "gpu_titanv=true"
    string qsubStrBasic = string(" qsub -b y -cwd ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " ";
    if (useGPU){
        qsubStrBasic +=  " -l " + gpuResouce+ " ";
    }

    qsubStrBasic += " -pe smp "+ to_string(numSlots) + " "
                  + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                  + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStrCmd = " " + cmdPath +" "+ cmdPara;

    string qsubStr = qsubStrBasic + " " + qsubStrCmd;
    int result = system(qsubStr.c_str());
    if (0 != result){
        cout<<qsubStr << "Error: qsub runs error"<<endl;
    }

    cout << "qsubTest submitted:" << jobName<<endl;
    cout<< "qsub program: "<< qsubStrCmd<<endl;
    return 0;

}
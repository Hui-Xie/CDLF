//
// Created by Hui Xie on 9/22/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
using namespace std;

string cmdPath = "/Users/hxie1/temp_release/Examples/TestSegmentation3D/TestSegmentation3D";
string cmdPara = "/Users/hxie1/temp_netParameters  /Users/hxie1/msd/Task07_Pancreas/CDLFData";

void printUsage(char* argv0){
    cout<<"Test 3DSegGAN in HPC"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <jobName> <Queue> <numSlots>"<<endl;
    cout<<"jobName: it specify the output file in the ~/temp_qsub directory."<<endl;
    cout<<"GPUQueue: COE-GPU, UI-DEVELOP, UI-GPU, INFORMATICS-GPU, INFORMATICS-HM-GPU, all.q;  Choose only one." <<endl;
    cout<<"CPUQueue: COE, UI-DEVELOP, UI-HM, UI-MPI, INFORMATICS, all.q;  Choose only one." <<endl;
    cout<<"numSlots: number of slots"<<endl;
}


int main(int argc, char *argv[]) {
    // notes: all command paramaeters have a space at front and at tail
    if (4 != argc){
        printUsage(argv[0]);
        return -1;
    }
    string jobName = argv[1];
    string queue = argv[2];
    int numSlots = atoi(argv[3]);


    string gpuResouce= "gpu_titanv=true" ; // "gpu_1080ti=true" or "gpu_titanv=true"
    string qsubStrBasic = string(" qsub -b y -cwd ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " "
                          //+ " -l " + gpuResouce+ " "
                          + " -pe smp "+ to_string(numSlots) + " "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStrCmd = " " + cmdPath +" "+ cmdPara;

    string qsubStr = qsubStrBasic + " " + qsubStrCmd;
    int result = system(qsubStr.c_str());
    if (0 != result){
        cout<<qsubStr << "runs error"<<endl;
    }

    //cout<<"GPU Resource: "<<gpuResouce<<endl;
    cout << "qsubTest submitted:" << jobName<<endl;
    return 0;

}
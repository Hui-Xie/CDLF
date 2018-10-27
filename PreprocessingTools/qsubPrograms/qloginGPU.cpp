//
// Created by Hui Xie on 9/29/18.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include <iostream>
using namespace std;

/*  verrify GPU resource
 *  in node: lspci | grep -i nvidia
 * */

void printUsage(char* argv0){
    cout<<"qlogin to apply GPU resource in HPC"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <GPUQueue> [numSlots]"<<endl;
    cout<<"GPUQueue: COE-GPU,UI-DEVELOP,UI-GPU;  Choose only one." <<endl;
    cout<<"numSlots: number of slots"<<endl;
}


int main(int argc, char *argv[]) {
    // notes: all command paramaeters have a space at front and at tail
    if (3 != argc){
        printUsage(argv[0]);
        return -1;
    }
    string queue = argv[1];
    int numSlots = atoi(argv[2]);

    string gpuResouce= "gpu_titanv=true" ; //  "gpu_1080ti=true" or "gpu_titanv=true"

    string jobName = "SessGPU";
    string qsubStrBasic = string(" qlogin ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " " //specify one of them: COE-GPU,UI-DEVELOP,UI-GPU
                          + " -l " + gpuResouce+ " "
                          + " -pe smp "+ to_string(numSlots) + " "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStr = qsubStrBasic;
    system(qsubStr.c_str());

    cout<<"GPU Resource: "<<gpuResouce<<endl;
    cout << "qlogin submitted: " << jobName <<endl;
    return 0;

}

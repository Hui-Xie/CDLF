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
    cout<<argv0<<" <jobName> <GPUQueue> <numSlots>"<<endl;
    cout<<"jobName: it specify the output file in the ~/temp_qsub directory."<<endl;
    cout<<"GPUQueue: COE-GPU, UI-DEVELOP, UI-GPU, INFORMATICS-GPU, INFORMATICS-HM-GPU, all.q;  Choose only one." <<endl;
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

    string gpuResouce= "gpu_titanv=true" ; //  "gpu_1080ti=true" or "gpu_titanv=true"

    string qsubStrBasic = string(" qlogin ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " "
                          + " -l " + gpuResouce+ " "
                          + " -pe smp "+ to_string(numSlots) + " "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStr = qsubStrBasic;
    int result = system(qsubStr.c_str());
    if (0 != result){
        cout<<qsubStr << "runs error"<<endl;
    }

    cout<<"GPU Resource: "<<gpuResouce<<endl;
    cout << "qlogin submitted: " << jobName <<endl;
    return 0;

}

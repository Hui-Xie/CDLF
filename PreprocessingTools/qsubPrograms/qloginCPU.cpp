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
    cout<<"qlogin to apply CPU resource in HPC"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <CPUQueue> [numSlots]"<<endl;
    cout<<"CPUQueue: COE,UI-DEVELOP,UI-HM, UI-MPI;  Choose only one." <<endl;
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

    string jobName = "SessCPU";
    string qsubStrBasic = string(" qlogin ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " " //specify one of them: COE,UI-DEVELOP,UI-HM, UI-MPI
                          + " -pe smp "+ to_string(numSlots) + " "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStr = qsubStrBasic;
    system(qsubStr.c_str());

    cout << "qlogin submitted: " << jobName <<endl;
    return 0;

}

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
    cout<<argv0<<" <jobName> <CPUQueue> <numSlots>"<<endl;
    cout<<"jobName: it specify the output file in the ~/temp_qsub directory."<<endl;
    cout<<"CPUQueue: COE, UI-DEVELOP, UI-HM, UI-MPI, all.q;  Choose only one." <<endl;
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

    string qsubStrBasic = string(" qlogin ")
                          + " -N " + jobName + " "
                          + " -q " + queue + " "
                          + " -pe smp "+ to_string(numSlots) + " "
                          + " -e ~/temp_qsub/Error_" + jobName + ".txt "
                          + " -o ~/temp_qsub/StdOutput_" + jobName + ".txt ";
    string qsubStr = qsubStrBasic;
    int result = system(qsubStr.c_str());
    if (0 != result){
        cout<<qsubStr << "runs error"<<endl;
    }

    cout << "qlogin submitted: " << jobName <<endl;
    return 0;

}

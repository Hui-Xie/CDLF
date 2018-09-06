//
// Created by Hui Xie on 9/6/18.
//
#include <iostream>


using namespace std;


void printUsage(char* argv0){
    cout<<"============= Uniform Image in Consistent Size and Spacing ==========="<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<fullPathFileName> <sizeX> <sizeY> <sizeZ> <spacingX> <spacingY> <spacingZ>"<<endl;
    cout<<endl;
}

int main(int argc, char *argv[]) {

    printUsage(argv[0]);
    if (argc != 8) {
        cout << "Error: the number of parameters is incorrect." << endl;
        return -1;
    }




    return 0;
}
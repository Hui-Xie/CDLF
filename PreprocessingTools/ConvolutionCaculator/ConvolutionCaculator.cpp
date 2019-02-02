
#include <iostream>
#include <Tools.h>

using namespace std;


void printUsage(char* argv0){
    cout<<"============= Compute OutputSize of Convolution  ==========="<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"  <PreTensorSize>  <FilterSize>  <Stride>  <NumFilters> "<<endl;
    cout<<"e.g.  ./ConvolutionCaculator 500*500*90 5*5*5  2*2*2 3"<<endl;
    cout<<endl;
}

int main(int argc, char *argv[]) {
    if (5 != argc) {
        cout << "Error: input parameter error" << endl;
        printUsage(argv[0]);
        return -1;
    }

    const string preTensorSizeStr = argv[1];
    const string filterSizeStr = argv[2];
    const string strideStr = argv[3];
    const int numFilters = atoi(argv[4]);


    const vector<int> inputTensorSize = str2Vector(preTensorSizeStr);
    const vector<int> filterSize = str2Vector(filterSizeStr);
    const vector<int> stride =  str2Vector(strideStr);

    cout<<"Info: it is a good practice to keep the output in each dimension odd."<<endl<<endl;

    const int dimFilter = filterSize.size();
    if (dimFilter != stride.size()) {
        cout << "Error: the dimension of filterSize and stride should be same." << endl;
        return -2;
    }

    if (!isElementBiggerThan0(stride)){
        cout<<"Error: the stride should be greater than 0. "<<endl;
        return -2;
    }

    int numInputFeatures  = 0;
    const int dimX = inputTensorSize.size();
    if (dimX == dimFilter) {
        numInputFeatures = 1;
    } else if (dimX - 1 == dimFilter) {
        numInputFeatures = inputTensorSize[0];
    } else {
        cout<< "Error: the dimensionon of filterSize should be equal with or 1 less than of the dimension of previous layer tensorSize."
            << endl;
        return -1;
    }

    const int s = (1 == numInputFeatures) ? 0 : 1; //indicate whether previousTensor includes feature dimension


    for (int i = 0; i < dimFilter; ++i) {
        if (0 == filterSize[i] % 2 && filterSize[i] !=inputTensorSize[i + s]) {
            cout << "Error: the filterSize in each dimension should be odd, "
                    "or if it is even, it must be same size of corresponding dimension of tensorSize of input X."
                 << endl;
            return -1;
        }
    }


    vector<int> outputTensorSize = inputTensorSize;
    for (int i = 0; i+s < dimX; ++i) {
        outputTensorSize[i+s] = (outputTensorSize[i+s] - filterSize[i]) / stride[i] + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }

    if (1 ==s){
        outputTensorSize[0] = 1;
    }

    if (1 != numFilters) {
        outputTensorSize.insert(outputTensorSize.begin(), numFilters);
    }
    deleteOnes(outputTensorSize);

    cout<<"Output TensorSize : "<<vector2Str(outputTensorSize) <<endl;

    return 0;
}

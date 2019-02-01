#include <iostream>
#include <Tools.h>

using namespace std;


void printUsage(char* argv0){
    cout<<"============= Compute OutputSize of TransposedConvolution  ==========="<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"  <PreTensorSize>  <FilterSize>  <Stride>  <NumFilters>"<<endl;
    cout<<"e.g.  ./TransConvolutionCaculator 53*77*77 5*5*5  2*2*2  1"<<endl;
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

    if (!isElementBiggerThan0(stride)){
        cout<<"Error: the stride of convolutionLayer should be greater than 0. "<<endl;
        return -2;
    }
    if (filterSize.size() != stride.size() || filterSize.size() != inputTensorSize.size()){
        cout<<"Error: InputTensorSize, filterSize and stride should have same dims"<<endl;
        return -2;
    }

    vector<int> outputTensorSize = inputTensorSize;
    const int dim = outputTensorSize.size();
    for (int i = 0; i < dim; ++i) {
        outputTensorSize[i] = (outputTensorSize[i] - 1) * stride[i] + filterSize[i];
    }
    if (1 != numFilters) {
        outputTensorSize.insert(outputTensorSize.begin(), numFilters);
    }
    deleteOnes(outputTensorSize);

    cout<<"Output TensorSize : "<<vector2Str(outputTensorSize) <<endl;

    return 0;
}
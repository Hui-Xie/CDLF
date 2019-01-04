
#include <iostream>
#include <Tools.h>

using namespace std;


void printUsage(char* argv0){
    cout<<"============= Compute OutputSize of Convolution  ==========="<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"  <PreTensorSize>  <FilterSize>  <NumFilters>  <Stride> "<<endl;
    cout<<"e.g.  ./ConvolutionCaculator 500*500*90 5*5*5 3 2"<<endl;
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
    const int numFilters = atoi(argv[3]);
    const int stride = atoi(argv[4]);

    if (stride <=0){
        cout<<"Error: the stride of convolutionLayer should be greater than 0. "<<endl;
        return -2;
    }


    const vector<int> inputTensorSize = str2Vector(preTensorSizeStr);
    const vector<int> filterSize = str2Vector(filterSizeStr);

    vector<int> outputTensorSize = inputTensorSize;
    const int dim = outputTensorSize.size();
    for (int i = 0; i < dim; ++i) {
        outputTensorSize[i] = (outputTensorSize[i] - filterSize[i]) / stride + 1;
        // ref formula: http://cs231n.github.io/convolutional-networks/
    }
    if (1 != numFilters) {
        outputTensorSize.insert(outputTensorSize.begin(), numFilters);
    }
    deleteOnes(outputTensorSize);

    cout<<"Output TensorSize : "<<vector2Str(outputTensorSize) <<endl;

    return 0;
}

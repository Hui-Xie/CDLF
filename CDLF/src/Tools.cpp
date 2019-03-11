//
// Created by Hui Xie on 7/16/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
#include "Tools.h"
#include "assert.h"
#include <iostream>
#include <ctime>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <Tools.h>
#include <math.h>


bool sameVector(const vector<int>& vec1, const vector<int>& vec2){
    if (vec1.size() != vec2.size()){
        return false;
    }
    else{
        int length = vec1.size();
        for (int i=0; i< length; ++i){
            if (vec1[i] != vec2[i]) return false;
        }
        return true;
    }
}

size_t length(const vector<int>& vec){
    int dim = vec.size();
    if (dim >0){
        size_t length=vec[0]*1l;
        for(int i =1; i< dim; ++i){
            length *= vec[i];
        }
        return length;
    }
    else{
        return 0;
    }
}

size_t length(const int N, const int array[]){
    if (N >0){
        size_t length=array[0]*1l;
        for(int i =1; i< N; ++i){
            length *= array[i];
        }
        return length;
    }
    else{
        return 0;
    }
}

bool sameLength(const vector<int>& vec1, const vector<int>& vec2){
    return (length(vec1) == length(vec2));
}

vector<int> reverseVector(const vector<int>& src){
    int size = src.size();
    vector<int> target;
    for(int i = size-1; i>=0; --i){
        target.push_back(src[i]);
    }
    return target;
}

vector<float> reverseVector(const vector<float>& src){
    int size = src.size();
    vector<float> target;
    for(int i = size-1; i>=0; --i){
        target.push_back(src[i]);
    }
    return target;
}

vector<int> operator+ (const vector<int>& left, const int offset){
    const int N = left.size();
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] += offset;
    }
    return result;
}



vector<int> operator+ (const vector<int>& left, const vector<int>& right){
    assert(left.size() == right.size());
    const int N = left.size();
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] += right[i];
    }
    return result;
}


vector<int> operator- (const vector<int>& left, const vector<int>& right){
    assert(left.size() == right.size());
    const int N = left.size();
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] -= right[i];
    }
    return result;
}

vector<int> operator- (const vector<int> &minuend, const int subtrahend) {
    const int N = minuend.size();
    vector<int> result(minuend);
    for (int i=0; i<N;++i){
        result[i] -= subtrahend;
    }
    return result;
}


vector<int> operator* (const vector<int>& left, const int factor){
    const int N = left.size();
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] *= factor;
    }
    return result;
}

vector<int> operator* (const vector<int>& left, const vector<int>& right){
    const int N = left.size();
    if (N != right.size()){
        cout<<"Error: multiplication of 2 vector has different size."<<endl;
        return left;
    }
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] *= right[i];
    }
    return result;
}

vector<int> operator/ (const vector<int>& left, const int divisor){
    const int N = left.size();
    vector<int> result = left;
    for (int i=0; i<N;++i){
        result[i] /= divisor;
    }
    return result;
}

bool operator<= (const vector<int>& left, const vector<int>& right){
    assert(left.size() == right.size());
    const int N = left.size();
    for (int i=0; i<N;++i){
        if(left[i] > right[i]) return false;
    }
    return true;
}

bool operator!= (const vector<int>& left, const vector<int>& right){
    return !sameVector(left, right);
}

bool operator== (const vector<int>& left, const vector<int>& right){
    return sameVector(left, right);
}

// delete 1 in the tensorSize when dim >2
void deleteOnes(vector<int>& vec){
    for (vector<int>::iterator it = vec.begin(); it!=vec.end();++it){
        if (1 == *it && vec.size() >2){
            it = vec.erase(it);
            --it;
        }
    }
}

int maxElement(const vector<int>& vec){
    const int N = vec.size();
    if (N <= 0){
        cout<<"Error: maxElement() must has non-zero size of vector as input."<<endl;
        return 0;
    }
    int max = vec[0];
    for (int i=1; i<N; ++i){
        if (vec[i] > max ) max = vec[i];
    }
    return max;
}

vector<int> nonZeroIndex(const vector<int>& vec){
    vector<int> result;
    const int N = vec.size();
    for (int i=0; i<N; ++i){
        if (0 != vec[i]) result.push_back(i);
    }
    return result;
}

void printVector(const vector<int>& vec){
    int N = vec.size();
    for (int i=0; i< N; ++i){
        std::cout<<vec[i]<<" ";
    }
    std::cout<<endl;

}

string vector2Str(const vector<int>& vec){
    int N= vec.size();
    string result ="{";
    for (int i=0;i< N; ++i){
        if (i != N-1){
            result += to_string(vec[i]) + "*";
        }
        else{
            result += to_string(vec[i]);
        }
     }
    result +="}";
    return result;
}

string array2Str(const int array[], const int N){
    string result ="{";
    for (int i=0;i< N; ++i){
        if (i != N-1){
            result += to_string(array[i]) + "*";
        }
        else{
            result += to_string(array[i]);
        }
    }
    result +="}";
    return result;
}

void vector2Array(const vector<int>& vec, int & N, int array[]){
    N = vec.size();
    array = new int [N];
    for (int i=0; i< N; ++i){
        array[i] = vec[i];
    }
}

vector<int> str2IntVector(const string &str){
    string tempStr = str;
    int N = tempStr.size();
    for (int i=0; i<N; ++i){
        if (!isdigit(tempStr[i])) tempStr[i] = ' ';
    }
    vector<int> vec;
    stringstream stream(tempStr);
    int num;
    while (stream >> num){
        vec.push_back(num);
    }
    return vec;
}

vector<float> str2FloatVector(const string &str){
    string tempStr = str;
    int N = tempStr.size();
    for (int i=0; i<N; ++i){
        if (!isdigit(tempStr[i]) && tempStr[i] != '.') tempStr[i] = ' ';
    }
    vector<float> vec;
    stringstream stream(tempStr);
    float num;
    while (stream >> num){
        vec.push_back(num);
    }
    return vec;
}

vector<int> generateRandomSequence(const int range) {
    vector<int> sequence;
    sequence.reserve(range);
    for (int i = 0; i < range; ++i) {
       sequence.push_back(i);
    }
    int M = range/2;
    srand (time(NULL));
    for (int i= 0; i<M; ++i){
        int r1 = rand() % range;
        int r2 = rand() % range;
        int temp = sequence[r1];
        sequence[r1] = sequence[r2];
        sequence[r2] = temp;
    }
    return sequence;
}

vector<int> generatePositiveNegativeRandomNumber(const int vectorSize, const int maxValue){
    vector<int> result(vectorSize, 0);
    if (maxValue <= 0 ){
        cout<<"Error: maxValue in generatePositiveNegativeRandomNumber should be greater than 0. "<<endl;
        return result;
    }
    srand (time(NULL));
    const int range = 2*maxValue +1;
    for (int i= 0; i<vectorSize; ++i){
        result[i] = rand() % range - maxValue;
    }
    return result;
}

vector<float> generatePositiveNegativeRandomRadian(const int vectorSize, const float maxRadian){
    vector<float> result(vectorSize, 0);
    if (maxRadian <= 0 ){
        cout<<"Error: maxRadian in generatePositiveNegativeRandomRadian should be greater than 0. "<<endl;
        return result;
    }
    srand (time(NULL));
    for (int i= 0; i<vectorSize; ++i){
        result[i] = -maxRadian + static_cast <float> (rand()) /static_cast <float> (RAND_MAX)* 2.0*maxRadian;
    }
    return result;
}

string getStemName(const string& filename){
    int pos = filename.rfind('.');
    return filename.substr(0, pos);

}

void printCurrentLocalTime(){
    time_t tt;
    time(&tt);
    tm TM = *localtime(&tt);
    printf("Current time: %4d-%02d-%02d %02d:%02d:%02d\n", 1900+TM.tm_year, TM.tm_mon+1, TM.tm_mday,TM.tm_hour, TM.tm_min,TM.tm_sec);
}

string getCurTimeStr(){
    time_t tt;
    time(&tt);
    tm TM = *localtime(&tt);
    char timeChar[20];
    sprintf(timeChar, "%4d%02d%02d_%02d:%02d:%02d", 1900+TM.tm_year, TM.tm_mon+1, TM.tm_mday,TM.tm_hour, TM.tm_min, TM.tm_sec);
    return string(timeChar);
}


vector<int> genDimsSpan(const vector<int> vec){
    const int N = vec.size();
    vector<int> dimsSpan(N, 1);
    for (int i= N-2; i>=0; --i){
        dimsSpan[i] = vec[i+1]*dimsSpan[i+1];
    }
    return dimsSpan;
}

string eraseAllSpaces(string str){
    for (string::iterator iter = str.begin(); iter< str.end(); ++iter){
          if (*iter == ' '){
              str.erase(iter);
              --iter;
          }
    }
    return str;
}

void dimA2SpanA(const int* dimA, const int N, int * spanA){
    if (N<1) return;
    spanA[N-1] = 1;
    for(int i=N-2; i>=0; --i){
        spanA[i] = spanA[i+1]*dimA[i+1];
    }
}

bool isElementBiggerThan0(const vector<int>& vec){
    const int N = vec.size();
    for(int i=0; i<N; ++i){
        if (vec[i]<=0) return false;
    }
    return true;
}

bool isElementEqual1(const vector<int>& vec){
    const int N = vec.size();
    for(int i=0; i<N; ++i){
        if (1 != vec[i]) return false;
    }
    return true;
}

bool isElementEqual0(const vector<float>& vec){
    const int N = vec.size();
    for(int i=0; i<N; ++i){
        if (0 != vec[i]) return false;
    }
    return true;
}

bool isContainSubstr(const string &str, const string subStr) {
    return (string::npos == str.find(subStr)) ? false : true;
}

// radianVec indicating rotating angle about axis 0,1,2;  R[3][4] are a rotation matrix without translation,
void getRotationMatrix(const vector<float> radianVec, double R[3][4]){

    const float a = radianVec[0];
    const float b = radianVec[1];
    const float c = radianVec[2];

    // translation =0
    R[0][3] = 0;
    R[1][3] = 0;
    R[2][3] = 0;

    R[0][0] = cos(b)*cos(c);
    R[0][1] = -cos(b)*sin(c);
    R[0][2] = sin(b);

    R[1][0] = sin(a)*sin(b)*cos(c)+ cos(a)*sin(c);
    R[1][1] = -sin(a)*sin(b)*sin(c)+ cos(a)*cos(c);
    R[1][2] = -sin(a)*cos(b);

    R[2][0] = -cos(a)*sin(b)*cos(c)+ sin(a)*sin(c);
    R[2][1] = cos(a)*sin(b)*sin(c)+ sin(a)*cos(c);
    R[2][2] = cos(a)*cos(b);
}

vector<int> getRotatedDims_UpdateTranslation(const vector<int> dims, double R[3][4]){
    const int N = dims.size();
    if (N != 3){
        cout<<"Error: Hinge Rotation needs dims.size == 3. Exit "<<endl;
        return vector<int>();
    }
    int maxx =0, maxy=0, maxz=0;
    int minx =0, miny=0, minz=0;
    
    for (int x= 0; x<dims[0]; x+=(dims[0]-1))
        for (int y=0; y<dims[1]; y += (dims[1]-1))
            for (int z=0; z<dims[2]; z += (dims[2]-1)){
                if (0 == x && 0 == y && 0 ==z)  continue;
                float x1 = R[0][0]*x+ R[0][1]*y + R[0][2]*z;
                float y1 = R[1][0]*x+ R[1][1]*y + R[1][2]*z;
                float z1 = R[2][0]*x+ R[2][1]*y + R[2][2]*z;

                maxx = (x1> maxx) ? x1 : maxx;
                maxy = (y1> maxy) ? y1 : maxy;
                maxz = (z1> maxz) ? z1 : maxz;

                minx = (x1< minx) ? x1 : minx;
                miny = (y1< miny) ? y1 : miny;
                minz = (z1< minz) ? z1 : minz;
            }

    R[0][3] = (minx<0) ? abs(minx) : 0;
    R[1][3] = (miny<0) ? abs(miny) : 0;
    R[2][3] = (minz<0) ? abs(minz) : 0;

    vector<int> rotatedDims(3,0);
    rotatedDims[0] = int(maxx- minx +1.5); // 0.5 is round; Size = maxIndex +1
    rotatedDims[1] = int(maxy- miny +1.5);
    rotatedDims[2] = int(maxz- minz +1.5);

    return rotatedDims;
}

void randomTranslate(vector<int>& vec, const int translationMaxValue){
    if ( 0 == translationMaxValue) {
        return;
    }
    else{
        const int N = vec.size();
        vector<int> drift = generatePositiveNegativeRandomNumber(N, translationMaxValue);
        for (int i =0 ;i< N; ++i){
            vec[i] += drift[i];
        }
    }
}

vector<int> getTopLeftIndexFrom(const vector<int> &imageDims, const vector<int> &subImageDims,
                                const vector<int>&  center) {
    if (!(subImageDims <= imageDims)){
        cout<<"Error: subImageDims should be less than imageDims."<<endl;
        std::exit(EXIT_FAILURE);
    }
    vector<int> topLeft;

    if (center.empty()){
        topLeft = (imageDims -subImageDims)/2;
    }
    else {
        topLeft = center - subImageDims/2;
    }

    for(int i=0; i<topLeft.size();++i){
        if (topLeft[i]+ subImageDims[i] > imageDims[i]){
            topLeft[i] = imageDims[i]- subImageDims[i];
        }
        if (topLeft[i] <0 ){
            topLeft[i] = 0;
        }
    }

    return topLeft;

}






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

int length(const vector<int>& vec){
    int dim = vec.size();
    if (dim >0){
        int length=vec[0];
        for(int i =1; i< dim; ++i){
            length *= vec[i];
        }
        return length;
    }
    else{
        return 0;
    }
}

int length(const int N, const int array[]){
    if (N >0){
        int length=array[0];
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

vector<int> str2Vector(const string& str){
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





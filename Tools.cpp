//
// Created by Sheen156 on 7/16/2018.
//
#include "Tools.h"
#include "assert.h"


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

vector<int> reverseVector(const vector<int>& src){
    int size = src.size();
    vector<int> target;
    for(int i = size-1; i>=0; --i){
        target.push_back(src[i]);
    }
    return target;
}


string vector2String(const vector<int>& src){
    string outputString;
    int length = src.size();
    for(int i=0; i< length; ++i){
        outputString += to_string(src[i])+ ((i==length-1)? " ": "*");
    }
    return outputString;
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

vector<int> operator* (const vector<int>& left, const int factor){
    const int N = left.size();
    vector<int> result(left);
    for (int i=0; i<N;++i){
        result[i] *= factor;
    }
    return result;
}



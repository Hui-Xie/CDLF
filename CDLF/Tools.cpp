//
// Created by Hui Xie on 7/16/2018.
//
#include "Tools.h"
#include "assert.h"
#include <iostream>
#include <ctime>


bool sameVector(const vector<long>& vec1, const vector<long>& vec2){
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

long length(const vector<long>& vec){
    long length=1;
    int dim = vec.size();
    for(int i =0; i< dim; ++i){
        length *= vec[i];
    }
    return length;
}

bool sameLength(const vector<long>& vec1, const vector<long>& vec2){
    return (length(vec1) == length(vec2));
}

vector<long> reverseVector(const vector<long>& src){
    int size = src.size();
    vector<long> target;
    for(int i = size-1; i>=0; --i){
        target.push_back(src[i]);
    }
    return target;
}


string vector2String(const vector<long>& src){
    string outputString;
    int length = src.size();
    for(int i=0; i< length; ++i){
        outputString += to_string(src[i])+ ((i==length-1)? " ": "*");
    }
    return outputString;
}

vector<long> operator+ (const vector<long>& left, const int offset){
    const int N = left.size();
    vector<long> result(left);
    for (int i=0; i<N;++i){
        result[i] += offset;
    }
    return result;
}

vector<long> operator+ (const vector<long>& left, const vector<long>& right){
    assert(left.size() == right.size());
    const int N = left.size();
    vector<long> result(left);
    for (int i=0; i<N;++i){
        result[i] += right[i];
    }
    return result;
}

vector<long> operator- (const vector<long>& left, const vector<long>& right){
    assert(left.size() == right.size());
    const int N = left.size();
    vector<long> result(left);
    for (int i=0; i<N;++i){
        result[i] -= right[i];
    }
    return result;
}

vector<long> operator- (const vector<long> &minuend, const int subtrahend) {
    const int N = minuend.size();
    vector<long> result(minuend);
    for (int i=0; i<N;++i){
        result[i] -= subtrahend;
    }
    return result;
}


vector<long> operator* (const vector<long>& left, const int factor){
    const int N = left.size();
    vector<long> result(left);
    for (int i=0; i<N;++i){
        result[i] *= factor;
    }
    return result;
}

vector<long> operator/ (const vector<long>& left, const int divisor){
    const int N = left.size();
    vector<long> result = left;
    for (int i=0; i<N;++i){
        result[i] /= divisor;
    }
    return result;
}

// delete 1 in the tensorSize when dim >2
void deleteOnes(vector<long>& vec){
    for (vector<long>::iterator it = vec.begin(); it!=vec.end();++it){
        if (1 == *it && vec.size() >2){
            it = vec.erase(it);
            --it;
        }
    }
}

vector<long> nonZeroIndex(const vector<long>& vec){
    vector<long> result;
    const int N = vec.size();
    for (int i=0; i<N; ++i){
        if (0 != vec[i]) result.push_back(i);
    }
    return result;
}

void printVector(const vector<long>& vec){
    int N = vec.size();
    for (int i=0; i< N; ++i){
        std::cout<<vec[i]<<" ";
    }
    std::cout<<endl;

}

string vector2Str(const vector<long>& vec){
    int N= vec.size();
    string result ="{ ";
    for (int i=0;i< N; ++i){
        if (i != N-1){
            result += to_string(vec[i]) + "*";
        }
        else{
            result += to_string(vec[i]);
        }
     }
    result +=" }";
    return result;
}

vector<long> generateRandomSequence(const long range) {
    vector<long> sequence;
    sequence.reserve(range);
    for (long i = 0; i < range; ++i) {
       sequence.push_back(i);
    }
    long M = range/2;
    srand (time(NULL));
    for (long i= 0; i<M; ++i){
        long r1 = rand() % range;
        long r2 = rand() % range;
        long temp = sequence[r1];
        sequence[r1] = sequence[r2];
        sequence[r2] = temp;
    }
    return sequence;
}






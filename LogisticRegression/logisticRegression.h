#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iostream>
#include <functional>
// #include<>
#include <algorithm>


struct Data{

    int label;
    std::vector<double> featVector;

};

struct Weights{


    std::vector<double> params;
    
};


class LogisticRegression{

      
        public:
            float lr;
            std::vector<Data> dataset;
            Weights w;
            std::vector<double> calculateGradient(Data &, double *);
            void doGradientDescent(std::vector<double> &);
            void doGradientDescentData(std::vector<Data> &,double *);
            void doGradientDescentData(Data &,double *);
            void train(int);
            
};


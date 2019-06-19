#include <vector>
#include <cmath>
// #include<>

struct Data{

    int label;
    std::vector<double> featVector;

};

struct Weights{


    std::vector<double> params;
    
    Weights(int n){

        for (int i=0;i<n;i++){
            params.push_back(0.0);
        }
    }


};


class LogisticRegression{

      
        public:
            float lr;
            std::vector<Data> dataset;
            Weights w(3);
            std::vector<double> calculateGradient(Data &, double *);
            void doGradientDescent(std::vector<double> &);
            void doGradientDescentData(std::vector<Data> &,double *);
            void doGradientDescentData(Data &,double *);
            void train(int);
            
};


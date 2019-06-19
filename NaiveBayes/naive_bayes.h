/* Given a finite set of labels the algorithm assigns labels to some input data,this algo assumes that   */
#include <map>
#include<vector>
#include<string>




struct data{

    int label; //class label index
    std::vector<int> val; //value in form of ine hot vector


};

class NaiveBayes {

    private:
        int numClasses = 0; // stores number of clases
        std::vector<data> trainData; //stores training data
        int featureVecsize; //stores size of feature vector
        std::map<std::string,int> labelmap ; //stores labels to index mapping
        std::map<int,double> classProbMap; //stores (P(Ck))
        std::map<int,std::vector<double> > classProbFeaturewise; //stores (P(xi/Ck))

    
    public:
        NaiveBayes(){
};
         
         void calculateProbs();
         void train();
         std::vector<double> Predict(std::vector<int>);
         void AddData(std::string ,std::vector<int> );
};
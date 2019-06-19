#include "logisticRegression.h"

std::vector<double> LogisticRegression::calculateGradient(Data &d, double *loss)
{
    // defining grad of logisitc function

    std::vector<double> gradVec;
    double r1 = std::inner_product(d.featVector.begin(), d.featVector.end(), LogisticRegression::w.params.begin(), 0) - d.label;
    *loss = r1 * r1;
    std::vector<double>::iterator it = LogisticRegression::w.params.begin();
    std::vector<double>::iterator it_d = d.featVector.begin();
    double grad_;
    while (it != LogisticRegression::w.params.end())
    {

        grad_ = *it * *it_d;
        gradVec.push_back(grad_ * r1) ;

        it++;
        it_d++;
    }

    return gradVec;
}

void LogisticRegression::doGradientDescent(std::vector<double> &gradVec)
{

    std::vector<double>::iterator it_w = LogisticRegression::w.params.begin();
    std::vector<double>::iterator it_g = gradVec.begin();

    while (it_w != w.params.end())
    {

        *it_w = *it_w - LogisticRegression::lr * *it_g;
        it_w++;
        it_g++;
    }
}

void LogisticRegression::doGradientDescentData(Data &d, double *l)
{

    std::vector<double> gradVecSum;
    std::vector<double> gradVec;
    // double loss = 0;
    gradVec = LogisticRegression::calculateGradient(d, l);
    // std::cout << "Loss in sample " << loss;
    doGradientDescent(gradVec);
}

void LogisticRegression::doGradientDescentData(std::vector<Data> &d, double *l)
{

    std::vector<double> gradVecSum;
    std::vector<Data>::iterator it_d = d.begin();
    std::vector<double> gradVec;
    *l = 0;
    double* loss;
    *loss = 0;

        while (it_d != d.end())
    {

        gradVec = calculateGradient(*it_d, loss);
        *l = *l + *loss;
        std::transform(gradVecSum.begin(), gradVecSum.end(), gradVec.begin(), gradVec.begin(), std::plus<double>());
        it_d++;
    }
    // std::cout << "Total loss in batch : " << loss_sum;
    doGradientDescent(gradVecSum);
}

void LogisticRegression::train(int steps)
{
    double* loss_epoch;
    double* loss_data;
    for (int i = 0; i < steps; i++)
    {
        *loss_epoch = 0 ;
        *loss_data = 0 ;
        std::vector<Data>::iterator data_iter = LogisticRegression::dataset.begin();
        while (data_iter != LogisticRegression::dataset.end())
        {

            LogisticRegression::doGradientDescent(*data_iter, loss_data);
            *loss_epoch = *loss_epoch + *loss_data;
            data_iter++;
        }

        std::cout << "Epoch " << i << "Loss: " << loss_epoch << "\n";
    }
}

int main()
{
    std::random_device rd;
    LogisticRegression lreg;
    Weights w;
    int sz = 3;
    {

        std::random_device rd; 
    // Mersenne twister PRNG, initialized with seed from previous random device instance
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(1, 2); 
        double sample;
        for (int i=0;i<sz;i++){
            sample = d(gen);
            w.params.push_back(sample);

        }
    }
    


    std::vector<double> r;
    int x1, x2, x3, y;
    Data d;

    for (int i = 0; i < 6; i++)
    {

        x1 = (float)rand() / RAND_MAX;
        x2 = (float)rand() / RAND_MAX;
        x3 = (float)rand() / RAND_MAX;
        r.push_back(x1);
        r.push_back(x2);
        r.push_back(x3);
        d.featVector = r;

        y = 2 * x1 + 3 * x2 + 4 * x3;
        d.label = y;

        lreg.dataset.push_back(d);
        r.clear();
    }

    lreg.train(5);

    return 0;
}



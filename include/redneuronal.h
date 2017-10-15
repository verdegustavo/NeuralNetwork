#ifndef REDNEURONAL_H
#define REDNEURONAL_H

#include <iostream>
#include <armadillo>
#include <string>
#include <sstream>
#include <ctime>
#include <fstream>
#include <vector>
#include <map>
#include "logFile.h"

class RedNeuronal {
    private:
        unsigned short _numCapas;
        float _costo, _epsilon;
        unsigned short _numEntradas;
        unsigned short _numHiddenNodes;
        unsigned short _numSalidas;
        arma::mat _weights1, _weights2, _weights1Grad, _weights2Grad;
        LogFile *_log;
        stringstream _inLog;

    protected:

    public:
        RedNeuronal(unsigned short cantCapas,
                    unsigned short cantEntradas,
                    unsigned short cantHidden,
                    unsigned short cantSalidas,
                    float ep);

        ~RedNeuronal();

        unsigned short GetnumCapas();
        arma::mat Getgradiente();
        float Getcosto();
        unsigned short GetnumEntradas();
        unsigned short GetnumSalidas();
        arma::mat transfer(const char * funcion,
                           arma::mat argumentos);
        arma::mat transferGrad(const char * funcion,
                            arma::mat argumentos);
        void calcCosto(float lambda,
                            arma::mat *X,
                            arma::mat *Y);
        void backpropagation(float lambda,
                            arma::mat *X,
                            arma::mat *Y);
        void aprender(arma::mat *X,
                      arma::mat *Y,
                      unsigned short algoritmo,
                      float alpha,
                      unsigned short numIter);
        arma::uvec predecir(arma::mat *X);
        void saltar(float learning_rate,
                    arma::mat *X,
                    arma::mat *Y,
                    arma::mat *W1,
                    arma::mat *W2,
                    arma::mat *Wg1,
                    arma::mat *Wg2);
        void guardarPesos(const char *path);
};

#endif // REDNEURONAL_H

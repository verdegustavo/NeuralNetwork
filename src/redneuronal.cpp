#include "redneuronal.h"

RedNeuronal::RedNeuronal(unsigned short cantCapas, unsigned short cantEntradas, unsigned short cantHidden, unsigned short cantSalidas, float ep) {
    //ctor
    _numCapas = cantCapas;
    _numEntradas = cantEntradas;
    _numHiddenNodes = cantHidden;
    _numSalidas = cantSalidas;
    _epsilon = ep;
    arma::arma_rng::set_seed(time(NULL));
    _weights1 = arma::randn<arma::mat>(_numHiddenNodes,_numEntradas + 1 );
    _weights2 = arma::randn<arma::mat>(_numSalidas,_numHiddenNodes + 1);
    _weights1 = _weights1*2*_epsilon - _epsilon;
    _weights2 = _weights2*2*_epsilon - _epsilon;
//    _weights1.load("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/W1.mat", arma::raw_ascii);
//    _weights2.load("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/W2.mat", arma::raw_ascii);
    try {
        _log = new LogFile("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/log/red_neuronal.log");
    } catch (...) {
        std::cout << "Hubo un problema al crear el archivo de log: " << std::endl;
    }

}


RedNeuronal::~RedNeuronal() {
    delete _log;
}


unsigned short RedNeuronal::GetnumCapas() {
    return _numCapas;
}


float RedNeuronal::Getcosto() {
    return _costo;
}


unsigned short RedNeuronal::GetnumEntradas() {
    return _numEntradas;
}


unsigned short RedNeuronal::GetnumSalidas() {
    return _numSalidas;
}

arma::mat RedNeuronal::transfer(const char * funcion, arma::mat argumentos) {
    if ( strncmp(funcion,"hardlim",7) == 0)
        return arma::conv_to<arma::mat>::from(argumentos >= 0);
    else  if ( strncmp(funcion,"purelin",7) == 0)
        return argumentos;
    else if (strncmp(funcion,"logsig",6) == 0)
        return 1/(1 + exp(-1*argumentos));
    else {
        std::cout << "WARNING: Funcion de transferencia no valida." << std::endl;
        return argumentos.zeros();
    }
}

arma::mat RedNeuronal::transferGrad(const char * funcion, arma::mat argumentos) {
/*    if ( strncmp(funcion,"hardlim",7) == 0)
        return arma::conv_to<arma::mat>::from(argumentos >= 0);
    else  if ( strncmp(funcion,"purelin",7) == 0)
        return argumentos;
    else */if (strncmp(funcion,"logsig",6) == 0)
        return transfer("logsig",argumentos)%(1 - transfer("logsig",argumentos));
    else {
        _inLog << "WARNING: Funcion de transferencia no valida.";
        _log->escribirLog(1,&_inLog);
        return argumentos.zeros();
    }
}

void RedNeuronal::calcCosto(float lambda, arma::mat *X, arma::mat *Y) {
    arma::mat a1, a2, a3, z2, z3, temp;
    float regW1, regW2;
    unsigned long m = X->n_rows;

    a1 = arma::join_horiz(arma::ones(m,1), *X); // a1 = [ones(m, 1) X];
    z2 = a1 * _weights1.t(); // Z2 = a1*Theta1';
    a2 = transfer("logsig",z2);
    a2 = arma::join_horiz(arma::ones(m,1), a2); // a2 = [ones(m, 1) a2];
    z3 = a2 * _weights2.t(); // Z3 = a2*Theta2';
    a3 = transfer("logsig",z3);

    temp = _weights1%_weights1;
    temp.col(0) = arma::zeros<arma::mat>(temp.n_rows,1);
    regW1 = arma::accu(temp);
    temp = _weights2%_weights2;
    temp.col(0) = arma::zeros<arma::mat>(temp.n_rows,1);
    regW2 = arma::accu(temp);

    _costo = (1/(float)m)*arma::accu( -1*(*Y)%arma::log(a3) - (1 - (*Y))%arma::log(1 - a3) ) + (lambda/(2*(float)m))*regW1 + (lambda/(2*(float)m))*regW2;
}

void RedNeuronal::backpropagation(float lambda, arma::mat *X, arma::mat *Y) {
    arma::mat a1, a2, a3, z2, z3, s2, s3, temp, DELTA_1, DELTA_2, W1_reg, W2_reg;
    float regW1, regW2;
    unsigned long m = X->n_rows;

// Calcular Costo:
    a1 = arma::join_horiz(arma::ones(m,1), *X); // a1 = [ones(m, 1) X];
    z2 = a1 * _weights1.t(); // Z2 = a1*Theta1';
    a2 = transfer("logsig",z2);
    a2 = arma::join_horiz(arma::ones(m,1), a2); // a2 = [ones(m, 1) a2];
    z3 = a2 * _weights2.t(); // Z3 = a2*Theta2';
    a3 = transfer("logsig",z3);

    temp = _weights1%_weights1;
    temp.col(0) = arma::zeros<arma::mat>(temp.n_rows,1);
    regW1 = arma::accu(temp);
    temp = _weights2%_weights2;
    temp.col(0) = arma::zeros<arma::mat>(temp.n_rows,1);
    regW2 = arma::accu(temp);

    _costo = (1/(float)m)*arma::accu( -1*(*Y)%arma::log(a3) - (1 - (*Y))%arma::log(1 - a3) ) + (lambda/(2*(float)m))*regW1 + (lambda/(2*(float)m))*regW2;


// Calcular Gradientes de los pesos:
    DELTA_1 = arma::zeros<arma::mat>(_weights1.n_rows, _weights1.n_cols);
    DELTA_2 = arma::zeros<arma::mat>(_weights2.n_rows, _weights2.n_cols);

    for(unsigned int i=0; i<m; ++i) {
        s3 = (a3.row(i) - Y->row(i)).t();
        s2 = (_weights2.cols(1,_weights2.n_cols - 1).t())*s3%(transferGrad("logsig",z2.row(i))).t();
        DELTA_1 =  DELTA_1 + s2*a1.row(i);
        DELTA_2 =  DELTA_2 + s3*a2.row(i);
    }

    W1_reg = _weights1;
    W1_reg.col(0) = arma::zeros<arma::mat>(W1_reg.n_rows,1);
    W2_reg = _weights2;
    W2_reg.col(0) = arma::zeros<arma::mat>(W2_reg.n_rows,1);

    _weights1Grad = (1/(float)m)*DELTA_1 + (lambda/(float)m)*W1_reg;
    _weights2Grad = (1/(float)m)*DELTA_2 + (lambda/(float)m)*W2_reg;
}

void RedNeuronal::aprender(arma::mat *X, arma::mat *Y,unsigned short algoritmo, float alpha, unsigned short numIter) {
    unsigned long m = X->n_rows;
    std::ofstream costoFile;
    try {
        costoFile.open("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/costo.mat", std::ofstream::out | std::ofstream::app);
    } catch (...) {
        std::cout << "ERROR! Hubo un problema al abrir el archivo de costos." << std::endl;
    }

    std::cout << "Entrenando..." << std::endl;
    arma::vec Wunrolled, WgUnrolled;

    switch (algoritmo) {
        case 0: // ************* STEEPEST DESCENT BACKPROPAGATION************* //
            std::cout << "Uitilizando SDBP..." << std::endl;
            for(unsigned short i = 0; i < numIter; ++i) {
                backpropagation(1,X,Y);
                Wunrolled = arma::join_vert( arma::vectorise(_weights1) , arma::vectorise(_weights2) );
                WgUnrolled = arma::join_vert(arma::vectorise(_weights1Grad) , arma::vectorise(_weights2Grad));
                Wunrolled = Wunrolled - alpha*(1/(float)m)*WgUnrolled;
                _weights1 = arma::reshape( Wunrolled(arma::span(0,_numHiddenNodes*(_numEntradas + 1)-1)) , _numHiddenNodes , _numEntradas + 1);
                _weights2 = arma::reshape( Wunrolled(arma::span(_numHiddenNodes*(_numEntradas + 1), Wunrolled.n_rows - 1)) , _numSalidas , _numHiddenNodes + 1);
                costoFile << _costo << std::endl;
            }
        break;

        case 1: /* ************* CONJUGATE GRADIENT BACKPROPAGATION *************
                    Método de Dirección de Búsqueda: Hestenes-Stiefel
                    Método de Línea de Búsqueda y Criterio de Tamaño de Paso: Golden Search Line
                */
            arma::mat W1init, W2init, W1GradInit, W2GradInit;
            backpropagation(1,X,Y);
            WgUnrolled = arma::join_vert(arma::vectorise(_weights1Grad) , arma::vectorise(_weights2Grad));
            W1init = _weights1;
            W2init = _weights2;
            W1GradInit = _weights1Grad;
            W2GradInit = _weights2Grad;

            // Paso 1: Seleccionar la primera dirección de búsqueda.
            arma::vec pAnterior = WgUnrolled;

            for (unsigned short k = 1; k < numIter; ++k) {
                // Interval Location:
                float lastCosto;
                std::map<float,float> mapIntLoc;
                std::map<float,float>::iterator it_mapa_intLoc = mapIntLoc.begin();

                float eps = 0.0375;
                mapIntLoc[eps] = _costo; // Para evitar problemas de memoria, se ingresa un registro inicial en el mapa.

                do {
                    lastCosto = _costo;
                    eps = 2*eps;

                    saltar(eps, X, Y, &W1init, &W2init, &W1GradInit, &W2GradInit);

                    mapIntLoc[eps] = _costo;
                    _inLog << "eps = " << eps << ", Costo = " << _costo;
                    _log->escribirLog(0,&_inLog);
                    if (mapIntLoc.size() > 3) {
                        it_mapa_intLoc = mapIntLoc.begin();
                        mapIntLoc.erase(it_mapa_intLoc);
                        it_mapa_intLoc = mapIntLoc.end();
                    }
                } while (_costo < lastCosto);

                float a, b, c, d;
                it_mapa_intLoc = mapIntLoc.end();
                --it_mapa_intLoc;
                b = it_mapa_intLoc->first;
                it_mapa_intLoc = mapIntLoc.begin();
                a = it_mapa_intLoc->first;

                if (mapIntLoc.size() == 3) {
                    ++it_mapa_intLoc;
                    mapIntLoc.erase(it_mapa_intLoc); // =====>>> OJO: SI EL MAPA QUEDA CON UN SOLO REGISTRO, ¿QUÉ PASA? ¡Problemas de Memoria!
                }
                it_mapa_intLoc = mapIntLoc.end();

                _inLog << "INTERVAL LOCATION: Valor de a = " << a << "\nINTERVAL LOCATION: Valor de b = " << b;
                _log->escribirLog(0,&_inLog);

                // Interval Reduction with Golden Search Line
                float tao = 0.618; // Definido por el algoritmo GSL
                c = a + (1 - tao)*(b - a);
                d = b - (1 - tao)*(b - a);

                saltar(c, X, Y, &W1init, &W2init, &W1GradInit, &W2GradInit);


                std::map<float,float> mapIntRed;
                std::map<float,float>::iterator it_mapa_intRed = mapIntRed.begin();
                it_mapa_intRed = mapIntRed.begin();
                mapIntRed[c] = _costo;

                _inLog << "INTERVAL REDUCTION: Valor de c = " << c << ", Costo = " << _costo ;
                _log->escribirLog(0,&_inLog);

                saltar(d, X, Y, &W1init, &W2init, &W1GradInit, &W2GradInit);

                mapIntRed[d] = _costo;

                _inLog << "INTERVAL REDUCTION: Valor de d = " << d << ", Costo = " << _costo ;
                _log->escribirLog(0,&_inLog);

                for (it_mapa_intLoc=mapIntLoc.begin(); it_mapa_intLoc!=mapIntLoc.end(); ++it_mapa_intLoc) {
                    _inLog << it_mapa_intLoc->first << " => " << it_mapa_intLoc->second << "\n" ;
                    _log->escribirLog(0,&_inLog);
                }
                for (it_mapa_intRed=mapIntRed.begin(); it_mapa_intRed!=mapIntRed.end(); ++it_mapa_intRed) {
                    _inLog << it_mapa_intRed->first << " => " << it_mapa_intRed->second << "\n" ;
                    _log->escribirLog(0,&_inLog);
}
                float evaluacion;
                do {
                    float Fc, Fd;
                    it_mapa_intRed=mapIntRed.begin();
                    Fc = it_mapa_intRed->second;
                    ++it_mapa_intRed;
                    Fd = it_mapa_intRed->second;

                    if (Fc < Fd) {
                      //a = a;
                        b = d;
                        d = c;
                        c = a + (1 - tao)*(b - a);
                        //  Paso 2: Tomar un salto
                        saltar(c, X, Y, &W1init, &W2init, &W1GradInit, &W2GradInit);

                        Fd = Fc;
                        Fc = _costo;
                    } else {
                        a = c;
                      //b = b;
                        c = d;
                        d = b - (1 - tao)*(b - a);
                        //  Paso 2: Tomar un salto
                        saltar(d, X, Y, &W1init, &W2init, &W1GradInit, &W2GradInit);

                        Fc = Fd;
                        Fd = _costo;
                    }

                    _inLog << "a = " << a << "\nb = " << b << "\nc = " << c << "\nd = " << d << "\n------------\n" ;
                    _log->escribirLog(0,&_inLog);

                    evaluacion = b - a;

                } while ( evaluacion > 0.1);

                _inLog << "------------\n------------\n------------\n" ;
                _log->escribirLog(0,&_inLog);


                // Paso 3: Seleccionar la nueva dirección de salto.
                backpropagation(1,X,Y);
                WgUnrolled = arma::join_vert(arma::vectorise(_weights1Grad) , arma::vectorise(_weights2Grad));

                float beta = as_scalar(WgUnrolled.t() * WgUnrolled) / as_scalar(pAnterior.t() * pAnterior);  // Escalar de dirección de búsqueda con método Hestenes-Stiefel.
                _inLog << "beta = " << beta;
                _log->escribirLog(0,&_inLog);
                pAnterior = WgUnrolled - beta*pAnterior; // Nueva dirección de búsqueda.
                W1GradInit = arma::reshape( pAnterior(arma::span(0,_numHiddenNodes*(_numEntradas + 1)-1)) , _numHiddenNodes , _numEntradas + 1);
                W2GradInit = arma::reshape( pAnterior(arma::span(_numHiddenNodes*(_numEntradas + 1), pAnterior.n_rows - 1)) , _numSalidas , _numHiddenNodes + 1);

                W1init = _weights1;
                W2init = _weights2;

                std::cout << "CGBP: # iteración: " << k << " ->   Costo = " << _costo << std::endl;
                costoFile << _costo << std::endl;
            }
            break;
    }
    try {
        costoFile.close();
    } catch (...) {
        std::cout << "¡ERROR!: El archivo costo.mat tuvo un problema y no pudo cerrarse correctamente." << std::endl;
    }
}

arma::uvec RedNeuronal::predecir(arma::mat *X) {
    arma::mat a1, a2, a3, z2, z3, Y;
    unsigned long m = X->n_rows;

    a1 = arma::join_horiz(arma::ones(m,1), *X); // a1 = [ones(m, 1) X];
    z2 = a1 * _weights1.t(); // Z2 = a1*Theta1';
    a2 = transfer("logsig",z2);
    a2 = arma::join_horiz(arma::ones(m,1), a2); // a2 = [ones(m, 1) a2];
    z3 = a2 * _weights2.t(); // Z3 = a2*Theta2';
    a3 = transfer("logsig",z3);

/*    arma::uvec pCoursera;
    pCoursera = arma::index_max(a3,1);
    pCoursera++;

    return pCoursera;*/
    return arma::index_max(a3,1);
}

void RedNeuronal::saltar(float learning_rate, arma::mat *X, arma::mat *Y, arma::mat *W1, arma::mat *W2, arma::mat *Wg1, arma::mat *Wg2) {

    arma::vec Wunrolled, WgUnrolled;

    Wunrolled = arma::join_vert( arma::vectorise(*W1) , arma::vectorise(*W2) );
    WgUnrolled = arma::join_vert(arma::vectorise(*Wg1) , arma::vectorise(*Wg2));

    Wunrolled = Wunrolled - learning_rate*WgUnrolled;

    _weights1 = arma::reshape( Wunrolled(arma::span(0,_numHiddenNodes*(_numEntradas + 1)-1)) , _numHiddenNodes , _numEntradas + 1);
    _weights2 = arma::reshape( Wunrolled(arma::span(_numHiddenNodes*(_numEntradas + 1), Wunrolled.n_rows - 1)) , _numSalidas , _numHiddenNodes + 1);

    calcCosto(1,X,Y);
}

void RedNeuronal::guardarPesos(const char *path) {
    stringstream concatenador;
    concatenador << path << "W1.mat";
    _weights1.save(concatenador ,arma::raw_ascii);

    concatenador << path << "W2.mat";
    _weights2.save(concatenador,arma::raw_ascii);
}


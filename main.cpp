#include <iostream>
#include <cstdlib>
#include <sstream>
#include "redneuronal.h"
#include "logFile.h"
#include <fstream>
#include <string>

int main(int argc, const char **argv) {
    // Carga de configuración y parámetros de la red neuronal:
    std::ifstream configs ("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/conf/network.conf", std::ifstream::in);
    std::string linea;

    unsigned short capas, entradas, ocultas, salidas, algoritmo, iter;
    float lambda, alpha;

    for (unsigned short i = 0; std::getline(configs,linea); ++i) {
       std::size_t encontrado = linea.find("=");
       std::string configurador;
       configurador = linea.substr(0,encontrado);
       float valor;
       valor = atof(linea.substr(encontrado + 2).c_str());
       if (configurador.compare("cantCapas ") == 0) {
          capas = (unsigned short)valor;
       } else if (configurador.compare("cantEntradas ") == 0) {
          entradas = (unsigned short)valor;
       } else if (configurador.compare("cantHidden ") == 0) {
          ocultas = (unsigned short)valor;
       } else if (configurador.compare("cantSalidas ") == 0) {
          salidas = (unsigned short)valor;
       } else if (configurador.compare("backpropagation_lambda ") == 0) {
          lambda = valor;
       } else if (configurador.compare("learning_algoritmo ") == 0) {
          algoritmo = (unsigned short)valor;
       } else if (configurador.compare("learning_alpha ") == 0) {
          alpha = valor;
       } else if (configurador.compare("learning_num_iteraciones ") == 0) {
          iter = (unsigned short)valor;
       }
    }

    // Configuracion de data:
    std::string parametro;
    const char *data_entrada;
    const char *data_clasificada;
    const char *directorio;

    for (unsigned short i = 1; i <= argc; ++i) {
       if (i != argc)
          parametro = argv[i];
       if ((parametro.compare("-e") == 0) or (parametro.compare("--entrada") == 0)) {
          std::cout << "Detectada entrada -e: " << argv[i+1] << std::endl;
          data_entrada = argv[i+1];
       } else if ((parametro.compare("-c") == 0) or (parametro.compare("--clasificada") == 0)) {
          std::cout << "Detectada entrada -c: " << argv[i+1] << std::endl;
          data_clasificada = argv[i+1];
       } else if ((parametro.compare("-d") == 0) or (parametro.compare("--directorio") == 0)) {
          std::cout << "Detectada entrada -d: " << argv[i+1] << std::endl;
          directorio = argv[i+1];
       }
    }

    // Estableciendo archivo a escribir el log:
    stringstream inLog;
    LogFile logger("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/log/clasificador.log");

    // Comenzando programa:
    inLog << "Inicializando red neuronal con pesos iniciales aleatorios...";
    logger.escribirLog(0,&inLog);

    RedNeuronal nn (capas, entradas, ocultas, salidas, 0.12);

    inLog << "Red neuronal creada satisactoriamente con " << nn.GetnumCapas() << " capas.";
    logger.escribirLog(0,&inLog);

    arma::mat X,y;

    inLog << "Cargando set de entrenamiento...";
    logger.escribirLog(0,&inLog);

    X.load(data_entrada, arma::raw_ascii);
    inLog << "Datos de entrada cargados con el archivo: " << data_entrada;
    logger.escribirLog(0,&inLog);

    y.load(data_clasificada, arma::raw_ascii);
    inLog << "Datos de clasificacion cargados con el archivo: " << data_clasificada;
    logger.escribirLog(0,&inLog);
/*
    X.load("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/X.dat", arma::raw_ascii);
    y.load("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/yMapped.mat", arma::raw_ascii);
    y.load("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/yMap.dat", arma::raw_ascii);
*/
    inLog << "Set de entrenamiento cargado normalmente.";
    logger.escribirLog(0,&inLog);

    inLog << "Calculando costo y gradientes regularizados de la red...";
    logger.escribirLog(0,&inLog);

    nn.backpropagation(lambda,&X,&y);

    inLog << "Backprogation culminado exitosamente!";
    logger.escribirLog(0,&inLog);

    std::cout << "El costo de la red ANTES del entrenamiento es: " << nn.Getcosto() << std::endl;

    arma::mat Xtest;
    Xtest = X.rows(0,4);

    std::cout << "La predicción ANTES del entrenamiento es:\n" << nn.predecir(&Xtest) << std::endl;

    inLog << "Entrenando red neuronal...";
    logger.escribirLog(0,&inLog);

    nn.aprender(&X,&y,algoritmo,alpha,iter);

    inLog << "Entrenamiento terminado!";
    logger.escribirLog(0,&inLog);

    std::cout << "El costo de la red DESPUES del entrenamiento es: " << nn.Getcosto() << std::endl;

    std::cout << "La predicción DESPUES del entrenamiento es:\n" << nn.predecir(&Xtest) << std::endl;

    nn.guardarPesos(directorio);
    inLog << "Pesos guardados en la ruta " << directorio;
    logger.escribirLog(0,&inLog);

    arma::uvec p;

    inLog << "Calculando predicción...";
    logger.escribirLog(0,&inLog);
    p = nn.predecir(&X);
    inLog << "Cálculo completado.";
    logger.escribirLog(0,&inLog);

    inLog << "Exportando predicción...";
    logger.escribirLog(0,&inLog);
    p.save("/home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/prediccion.mat",arma::raw_ascii);
    inLog << "Predicción exportada a la ruta /home/gustavo/Programming/C++/CodeBlocks Projects/Red Neuronal/data/prediccion.mat";
    logger.escribirLog(0,&inLog);

    return 0;
}

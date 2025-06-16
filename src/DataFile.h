#ifndef _DATA_FILE_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <regex>

class DataFile
{
private:
    //nom du fichier
    std::string _nomDeFichier;

    //parametre global
    int _cas;
    int _nbImg; 
    int _nbPts;
    std::string _resultatChemin;
    int _debug;

    //parametre maillage
    int _Nx;
    double _xmin, _xmax;
    double _Tmax;
    double _cfl;

    //parametre physique
    double _gamma;
    double _kappa;
    double _nu;
    double _epsilon;
    double _r;
    double _lambda;

public:
    //constructeur et destructeur
    DataFile(std::string nomDeFichier);

    ~DataFile() = default;

    void litLeFichier();

    //on recupere les variables privées
    const std::string &getNomFichier() const { return _nomDeFichier; };
    int getCas() const { return _cas; };
    int getNbImg() const {return _nbImg; };
    int getNbPts() const {return _nbPts; };
    const std::string &getResultatChemin() const { return _resultatChemin; };
    int getNx() const { return _Nx; };
    double getXMin() const { return _xmin; };
    double getXMax() const { return _xmax; };
    double getTmax() const { return _Tmax; };
    double getCFL() const {return _cfl; };
    double getGamma() const {return _gamma; };
    double getKappa() const {return _kappa; };
    double getNu() const {return _nu; };
    double getEpsilon() const { return _epsilon; };
    double getR() const {return _r; };
    double getLambda() const {return _lambda; };
    double getDebug() const {return _debug; };

    //on essaie de changer la valeur de Nx
    void changeNx(int Nx) {_Nx = Nx;};
};

#define _DATA_FILE_H
#endif
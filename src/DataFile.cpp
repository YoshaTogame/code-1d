#include "DataFile.h"

DataFile::DataFile(std::string nomDeFichier) : _nomDeFichier(nomDeFichier)
{
}

// on lit le fichier des donnees
void DataFile::litLeFichier()
{
    std::ifstream data_file(_nomDeFichier.data());
    // on verifie qu on a reussi a ouvrir le fichier
    if (!data_file.is_open())
    {
        std::cout << "le fichier des donnees ne s est pas bien ouvert" << std::endl;
        abort();
    }

    std::string line;
    // on lit les parametres dans le fichier
    #if 1
    while (!data_file.eof())
    {
        getline(data_file, line);
        if (line.find("#") != std::string::npos)
        {
            // Ignore this line (comment)
        }
        else
        {
            if (line.find("Cas") != std::string::npos)
                data_file >> _cas;
            if (line.find("nbImg") != std::string::npos)
                data_file >> _nbImg;
            if (line.find("nbPts") != std::string::npos)
                data_file >> _nbPts;
            if (line.find("Chemin") != std::string::npos)
                data_file >> _resultatChemin;
            if (line.find("Debug") != std::string::npos)
                data_file >> _debug;    
            if (line.find("Nx") != std::string::npos)
                data_file >> _Nx;
            if (line.find("xmin") != std::string::npos)
                data_file >> _xmin;
            if (line.find("xmax") != std::string::npos)
                data_file >> _xmax;
            if (line.find("tmax") != std::string::npos)
                data_file >> _Tmax;
            if (line.find("cfl") != std::string::npos)
                data_file >> _cfl;
            if (line.find("gamma") != std::string::npos)
                data_file >> _gamma;
            if (line.find("kappa") != std::string::npos)
                data_file >> _kappa;
            if (line.find("nu") != std::string::npos)
                data_file >> _nu;
            if (line.find("epsilon") != std::string::npos)
                data_file >> _epsilon;
            if (line.find("r") != std::string::npos)
                data_file >> _r;
            if (line.find("lambda") != std::string::npos)
                data_file >> _lambda;
        }
    }
    #else
    while(!data_file.eof())
    {
        data_file >> _cas;
        data_file >> _nbImg;
        data_file >> _nbPts;
        data_file >> _resultatChemin;
        data_file >> _Nx;
        data_file >> _xmin;
        data_file >> _xmax;
        data_file >> _Tmax;
        data_file >> _cfl;
        data_file >> _gamma;
        data_file >> _kappa;
        data_file >> _nu;
        data_file >> _epsilon;
        data_file >> _r;
        data_file >> _lambda;
    }
    #endif
    //on creer le fichier resultat demandé par _resultatChemin
    int sys;
    sys = system(("mkdir -p ./" +_resultatChemin).c_str());
    sys = system(("rm -f ./*.dat" +_resultatChemin).c_str());
}
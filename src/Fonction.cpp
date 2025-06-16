#include <iostream>
#include "Fonction.h"
#include "Dense"
#include "Sparse"
#include "DataFile.h"
#include <cmath>
#include <list>
#include <iomanip>

using namespace Eigen;

void Build_Matrix(Eigen::SparseMatrix<double> &M, DataFile *df, double dt, Eigen::VectorXd vecRho)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double nu(df->getNu());
    double kappa(df->getKappa());
    double epsilon(df->getEpsilon());
    double r(df->getR());
    double dx = (xmax-xmin)/Nx;
    double lambda = dt/(dx*dx);
    double tempo;
    double cstB = 0.5*lambda*(epsilon-4*nu*nu*kappa*(1.-kappa)/epsilon);
    double cstB2 = 0.5*lambda*epsilon;
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    //on construit la matrice ligne par ligne dans le cas générale
    if (cas==1){
        tripletList.reserve(12*Nx);
        //premier bloc
        //premiere ligne
        //bloc 1
        tempo = (1.+r*dt + lambda*nu*(3.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(0,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(0,1,tempo));
        //bloc 2
        tempo = cstB*(8.+vecRho[0]+vecRho[1])/vecRho[0];
        tripletList.push_back(T(0,Nx,tempo));
        tempo =-cstB*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(0,Nx+1,tempo));
        //les lignes intermediaires
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(i,i-1,tempo));
            tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(i,i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(i,i+1,tempo));
            //second bloc
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(i,Nx+i-1,tempo));
            tempo = cstB*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(i,Nx+i,tempo));
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(i,Nx+i+1,tempo));
        }
        //derniere ligne des premiers blocs
        //premier bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(Nx-1,Nx-2,tempo));
        tempo = (1.+r*dt + lambda*nu*(3.+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(Nx-1,2*Nx-2,tempo));
        tempo = cstB*(4.+vecRho[Nx-1]+vecRho[Nx-2])/vecRho[Nx-1];
        tripletList.push_back(T(Nx-1,2*Nx-1,tempo));
        //premiere ligne des seconds blocs
        //bloc 2
        //permier bloc
        tempo = -cstB2*(8.+vecRho[0]+vecRho[1])/vecRho[0];
        tripletList.push_back(T(Nx,0,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(Nx,1,tempo));
        //second bloc
        tempo = (1. + lambda*nu*(3.+0.*(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(Nx,Nx,tempo));
        tempo = -lambda*nu*(1. + 0.*(1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(Nx,Nx+1,tempo));
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(Nx+i,i-1,tempo));
            tempo = -cstB2*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(Nx+i,i,tempo));
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(Nx+i,i+1,tempo));
            //dernier bloc
            tempo = -lambda*nu*(1. + 0.*(1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(Nx+i,Nx+i-1,tempo));
            tempo = (1. + lambda*nu*(2+0.*(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(Nx+i,Nx+i,tempo));
            tempo = -lambda*nu*(1 + 0.*(1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(Nx+i,Nx+i+1,tempo));
        }
        //premier bloc
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(2*Nx-1,Nx-2,tempo));
        //tempo = -cstB2*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tempo = cstB2*(4.+vecRho[Nx-1]+vecRho[Nx-2])/vecRho[Nx-1];
        tripletList.push_back(T(2*Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -lambda*nu*(1. + 0.*(1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(2*Nx-1,2*Nx-2,tempo));
        tempo = (1. + lambda*nu*(3.+0.*(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(2*Nx-1,2*Nx-1,tempo));
        //on fini de definir la matrice
        M.setFromTriplets(tripletList.begin(),tripletList.end());
    }
    else if (cas==4) {
        tripletList.reserve(12*Nx);
        //premier bloc
        //premiere ligne
        //bloc 1
        tempo = (1.+r*dt + lambda*nu*(3.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(0,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(0,1,tempo));
        //bloc 2
        tempo = cstB*(8.+vecRho[0]+vecRho[1])/vecRho[0];
        tripletList.push_back(T(0,Nx,tempo));
        tempo =-cstB*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(0,Nx+1,tempo));
        //les lignes intermediaires
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(i,i-1,tempo));
            tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(i,i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(i,i+1,tempo));
            //second bloc
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(i,Nx+i-1,tempo));
            tempo = cstB*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(i,Nx+i,tempo));
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(i,Nx+i+1,tempo));
        }
        //derniere ligne des premiers blocs
        //premier bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(Nx-1,Nx-2,tempo));
        tempo = (1.+r*dt + lambda*nu*(3.+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(Nx-1,2*Nx-2,tempo));
        tempo = cstB*(4.+vecRho[Nx-1]+vecRho[Nx-2])/vecRho[Nx-1];
        tripletList.push_back(T(Nx-1,2*Nx-1,tempo));
        //premiere ligne des seconds blocs
        //bloc 2
        //permier bloc
        tempo = -cstB2*(8.+vecRho[0]+vecRho[1])/vecRho[0];
        tripletList.push_back(T(Nx,0,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(Nx,1,tempo));
        //second bloc
        tempo = (1. + lambda*nu*(3.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(Nx,Nx,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(Nx,Nx+1,tempo));
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(Nx+i,i-1,tempo));
            tempo = -cstB2*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(Nx+i,i,tempo));
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(Nx+i,i+1,tempo));
            //dernier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(Nx+i,Nx+i-1,tempo));
            tempo = (1. + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(Nx+i,Nx+i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(Nx+i,Nx+i+1,tempo));
        }
        //premier bloc
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(2*Nx-1,Nx-2,tempo));
        //tempo = -cstB2*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tempo = cstB2*(4.+vecRho[Nx-1]+vecRho[Nx-2])/vecRho[Nx-1];
        tripletList.push_back(T(2*Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(2*Nx-1,2*Nx-2,tempo));
        tempo = (1. + lambda*nu*(3.+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(2*Nx-1,2*Nx-1,tempo));
        //on fini de definir la matrice
        M.setFromTriplets(tripletList.begin(),tripletList.end());
    }
    else {
        tripletList.reserve(12*Nx);
        //premier bloc
        //premiere ligne
        //bloc 1
        tempo = (1.+r*dt + lambda*nu*(2.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(0,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(0,1,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[Nx-1]));
        tripletList.push_back(T(0,Nx-1,tempo));
        //bloc 2
        tempo = cstB*(2.+(vecRho[1]+vecRho[Nx-1])/vecRho[0]);
        tripletList.push_back(T(0,Nx,tempo));
        tempo =-cstB*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(0,Nx+1,tempo));
        tempo = -cstB*(1.+vecRho[0]/vecRho[Nx-1]);
        tripletList.push_back(T(0,2*Nx-1,tempo));
        //les lignes intermediaires
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(i,i-1,tempo));
            tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(i,i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(i,i+1,tempo));
            //second bloc
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(i,Nx+i-1,tempo));
            tempo = cstB*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(i,Nx+i,tempo));
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(i,Nx+i+1,tempo));
        }
        //derniere ligne des premiers blocs
        //premier bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[0]));
        tripletList.push_back(T(Nx-1,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(Nx-1,Nx-2,tempo));
        tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[0]);
        tripletList.push_back(T(Nx-1,Nx,tempo));
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(Nx-1,2*Nx-2,tempo));
        tempo = cstB*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tripletList.push_back(T(Nx-1,2*Nx-1,tempo));
        //premiere ligne des seconds blocs
        //bloc 2
        //permier bloc
        tempo = -cstB2*(2.+(vecRho[1]+vecRho[Nx-1])/vecRho[0]);
        tripletList.push_back(T(Nx,0,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(Nx,1,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[Nx-1]);
        tripletList.push_back(T(Nx,Nx-1,tempo));
        //second bloc
        tempo = (1. + lambda*nu*(2.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(Nx,Nx,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(Nx,Nx+1,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[Nx-1]));
        tripletList.push_back(T(Nx,2*Nx-1,tempo));
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(Nx+i,i-1,tempo));
            tempo = -cstB2*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(Nx+i,i,tempo));
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(Nx+i,i+1,tempo));
            //dernier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(Nx+i,Nx+i-1,tempo));
            tempo = (1. + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(Nx+i,Nx+i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(Nx+i,Nx+i+1,tempo));
        }
        //premier bloc
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[0]);
        tripletList.push_back(T(2*Nx-1,0,tempo));
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(2*Nx-1,Nx-2,tempo));
        tempo = -cstB2*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tripletList.push_back(T(2*Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[0]));
        tripletList.push_back(T(2*Nx-1,Nx,tempo));
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(2*Nx-1,2*Nx-2,tempo));
        tempo = (1. + lambda*nu*(2+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(2*Nx-1,2*Nx-1,tempo));
        //on fini de definir la matrice
        M.setFromTriplets(tripletList.begin(),tripletList.end());
    }
/*    else {
        tripletList.reserve(12*Nx);
        //premier bloc
        //premiere ligne
        //bloc 1
        tempo = (1.+r*dt + lambda*nu*(2.+(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(0,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(0,1,tempo));
        tempo = -lambda*nu*(1. + (1.-2*kappa)*(vecRho[0]/vecRho[Nx-1]));
        tripletList.push_back(T(0,Nx-1,tempo));
        //bloc 2
        tempo = cstB*(2.+(vecRho[1]+vecRho[Nx-1])/vecRho[0]);
        tripletList.push_back(T(0,Nx,tempo));
        tempo =-cstB*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(0,Nx+1,tempo));
        tempo = -cstB*(1.+vecRho[0]/vecRho[Nx-1]);
        tripletList.push_back(T(0,2*Nx-1,tempo));
        //les lignes intermediaires
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = -lambda*nu*(1. + (1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(i,i-1,tempo));
            tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(i,i,tempo));
            tempo = -lambda*nu*(1 + (1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(i,i+1,tempo));
            //second bloc
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(i,Nx+i-1,tempo));
            tempo = cstB*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(i,Nx+i,tempo));
            tempo = -cstB*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(i,Nx+i+1,tempo));
        }
        //derniere ligne des premiers blocs
        //premier bloc
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[0]));
        tripletList.push_back(T(Nx-1,0,tempo));
        tempo = -lambda*nu*(1. + (1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(Nx-1,Nx-2,tempo));
        tempo = (1.+r*dt + lambda*nu*(2+(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[0]);
        tripletList.push_back(T(Nx-1,Nx,tempo));
        tempo = -cstB*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(Nx-1,2*Nx-2,tempo));
        tempo = cstB*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tripletList.push_back(T(Nx-1,2*Nx-1,tempo));
        //premiere ligne des seconds blocs
        //bloc 2
        //permier bloc
        tempo = -cstB2*(2.+(vecRho[1]+vecRho[Nx-1])/vecRho[0]);
        tripletList.push_back(T(Nx,0,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[1]);
        tripletList.push_back(T(Nx,1,tempo));
        tempo = cstB2*(1.+vecRho[0]/vecRho[Nx-1]);
        tripletList.push_back(T(Nx,Nx-1,tempo));
        //second bloc
        tempo = (1. + lambda*nu*(2.+0.*(1.-2*kappa)*(vecRho[1]+vecRho[Nx-1])/vecRho[0]));
        tripletList.push_back(T(Nx,Nx,tempo));
        tempo = -lambda*nu*(1. + 0.*(1.-2*kappa)*(vecRho[0]/vecRho[1]));
        tripletList.push_back(T(Nx,Nx+1,tempo));
        tempo = -lambda*nu*(1. + 0.*(1.-2*kappa)*(vecRho[0]/vecRho[Nx-1]));
        tripletList.push_back(T(Nx,2*Nx-1,tempo));
        for (int i = 1; i<Nx-1; i++){
            //premier bloc
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i-1]);
            tripletList.push_back(T(Nx+i,i-1,tempo));
            tempo = -cstB2*(2.+(vecRho[i-1]+vecRho[i+1])/vecRho[i]);
            tripletList.push_back(T(Nx+i,i,tempo));
            tempo = cstB2*(1.+ vecRho[i]/vecRho[i+1]);
            tripletList.push_back(T(Nx+i,i+1,tempo));
            //dernier bloc
            tempo = -lambda*nu*(1. + 0.*(1-2*kappa)*(vecRho[i]/vecRho[i-1]));
            tripletList.push_back(T(Nx+i,Nx+i-1,tempo));
            tempo = (1. + lambda*nu*(2+0.*(1.-2*kappa)*(vecRho[i+1]+vecRho[i-1])/vecRho[i]));
            tripletList.push_back(T(Nx+i,Nx+i,tempo));
            tempo = -lambda*nu*(1 + 0.*(1.-2.*kappa)*(vecRho[i]/vecRho[i+1]));
            tripletList.push_back(T(Nx+i,Nx+i+1,tempo));
        }
        //premier bloc
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[0]);
        tripletList.push_back(T(2*Nx-1,0,tempo));
        tempo = cstB2*(1.+ vecRho[Nx-1]/vecRho[Nx-2]);
        tripletList.push_back(T(2*Nx-1,Nx-2,tempo));
        tempo = -cstB2*(2.+(vecRho[0]+vecRho[Nx-2])/vecRho[Nx-1]);
        tripletList.push_back(T(2*Nx-1,Nx-1,tempo));
        //second bloc
        tempo = -lambda*nu*(1. + 0.*(1.-2.*kappa)*(vecRho[Nx-1]/vecRho[0]));
        tripletList.push_back(T(2*Nx-1,Nx,tempo));
        tempo = -lambda*nu*(1. + 0.*(1.-2.*kappa)*(vecRho[Nx-1]/vecRho[Nx-2]));
        tripletList.push_back(T(2*Nx-1,2*Nx-2,tempo));
        tempo = (1. + lambda*nu*(2+0.*(1.-2*kappa)*(vecRho[Nx-2]+vecRho[0])/vecRho[Nx-1]));
        tripletList.push_back(T(2*Nx-1,2*Nx-1,tempo));
        //on fini de definir la matrice
        M.setFromTriplets(tripletList.begin(),tripletList.end());
    } */
}

void BuildLaplacian(Eigen::SparseMatrix<double> &D, DataFile *df, double dt)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    double epsilon(df->getEpsilon());
    double lambda = dt/(dx*dx);
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    //on construit la matrice ligne par ligne dans le cas générale
    tripletList.reserve(3*Nx);
    double tempo0;
    double tempo1;
    tempo1 = 1. + 2.*kappa*nu*lambda;
    double tempo2;
    tempo2 = -kappa*lambda*nu;
    //std::cout << tempo1 << tempo2 << std::endl;
    if (cas==1 || cas==7){
        //conditions de Dirichlet pour rho
        tempo0 = (1+3.*kappa*nu*lambda);
        //premiere ligne
        tripletList.push_back(T(0,0,tempo0));
        tripletList.push_back(T(0,1,tempo2));
        //lignes intermediaire
        for(int i=1; i<Nx-1; i++){
            tripletList.push_back(T(i,i-1,tempo2));
            tripletList.push_back(T(i,i,tempo1));
            tripletList.push_back(T(i,i+1,tempo2));
        }
        //derniere ligne
        tripletList.push_back(T(Nx-1,Nx-2,tempo2));
        tripletList.push_back(T(Nx-1,Nx-1,tempo0));
        D.setFromTriplets(tripletList.begin(),tripletList.end());
    }
    else {
        //premiere ligne
        tripletList.push_back(T(0,0,tempo1));
        tripletList.push_back(T(0,1,tempo2));
        tripletList.push_back(T(0,Nx-1,tempo2));
        //ligne intermediaire
        for(int i=1; i<Nx-1; i++){
            tripletList.push_back(T(i,i-1,tempo2));
            tripletList.push_back(T(i,i,tempo1));
            tripletList.push_back(T(i,i+1,tempo2));
        }
        //derniere ligne
        tripletList.push_back(T(Nx-1,0,tempo2));
        tripletList.push_back(T(Nx-1,Nx-2,tempo2));
        tripletList.push_back(T(Nx-1,Nx-1,tempo1));
        D.setFromTriplets(tripletList.begin(),tripletList.end());
    }
}

void Initialize(Eigen::VectorXd &vecU, DataFile *df )
{
    int Nx(df->getNx());
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double x;
    double epsilon(df->getEpsilon());
    double nu(df->getNu());
    double kappa(df->getKappa());
    Eigen::VectorXd rho(Nx);    
    Eigen::VectorXd w(Nx);
    Eigen::VectorXd v(Nx);
    if (cas==0)
    {
        //cas test du soliton
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho[i] = 1.5 - 0.5/(pow(cosh(sqrt(0.5)*x),2));
            vecU[i] = rho[i];
            w[i] = 2. - 1.5/rho[i];
            vecU[i+Nx] = rho[i]*w[i];
            v[i] = 2.*pow(0.5,1.5)*sinh(sqrt(0.5)*x)/pow(cosh(sqrt(0.5)*x),3);
            vecU[i+2*Nx] = epsilon*v[i];
        }
    }
    else if (cas==1)
    {
        //probleme de Riemann dispersif
        double invdelta = 1000;
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho[i] = 1.5 - 0.5*tanh(invdelta*x);
            vecU[i] = rho[i];
            w[i] = 0.;
            vecU[i+Nx] = rho[i]*w[i];
            v[i] = -0.5*invdelta*(1.-pow(tanh(invdelta*x),2));
            vecU[i+2*Nx] = epsilon*v[i];
        }
        #if 0
        //on refait le cas de facon discontinu
        for (int i=0; i<Nx; i++){
            x = xmin + 0.5*dx + i*dx;
            if (x<0){
                rho[i] = 2.;
            }
            else {
                rho[i] = 1.;
            }
            rho[i] = 1.5 - 0.5*tanh(invdelta*x);
            w[i] = 0.;
            v[i] = 0.;
            //v[i] = -0.5*10.*(1.-pow(tanh(10.*x),2));
            vecU[i] = rho[i];
            vecU[i+Nx]=rho[i]*w[i];
            vecU[i+2.*Nx] = epsilon*rho[i]*v[i];
            //vecU[i+2.*Nx] = epsilon*v[i];
        }
        #endif
    }
    else if (cas==2){
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho[i] = 2.+sin(0.2*M_PI*x);
            vecU[i] = rho[i];
            w[i] = 1.;
            vecU[i+Nx] = rho[i]*w[i];
            v[i] = 0.2*M_PI*cos(0.2*M_PI*x);
            vecU[i+2*Nx] = epsilon*v[i];
        }
    }
    else if (cas==3){
        double pic = 1./(sqrt(M_PI));
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho[i] = pic*exp(-pow(x,2));
            v[i] = -2.*epsilon*pic*x;
            w[i] = 0.+2.*kappa*nu/epsilon*v[i];
            vecU[i] = rho[i];
            vecU[i+Nx] = rho[i]*w[i];
            vecU[i+2*Nx] = rho[i]*v[i];
        }
    }
    else if (cas==4 || cas==5 || cas==6 || cas==7 ||cas==8||cas==9 || cas==10){
        //cas test avec terme source repris du cas 2d, le cas 4 test avec des conditions de bords particuliere tandis que le cas 5 est periodique
        //cas 6 represente le cas avec variation de u et de r > 0
        //cas 7 -> cas 6 avec conditions limites spécifiques
        //cas 8 -> cas 7 avec ajout de lequation de poisson
        double xp;
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho[i] = rho0(0,x,df);
            v[i] = v0(0,x,df);
            w[i] = w0(0,x,df);
            vecU[i] = rho[i];
            vecU[i+Nx] = rho[i]*w[i];
            vecU[i+2*Nx] = rho[i]*v[i];
        }
    }
}

double rho0(double t,double x,DataFile *df)
{
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double xp;
    double c=1.;
    double sigma = 1.;
    double res;
    xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
    if (cas==4||cas==5||cas==6||cas==7||cas==8){
        res = 1.+exp(-0.5*pow(xp,2));
    }
    else if (cas==9){
        res = 1.;
    }
    else if (cas==10){
        double tau = 2.*M_PI;
        res = 2.+sin(tau*x);
        //res = 1.;
    }
    else{
        //std::cout << "ce cas n'est pas défini" << std::endl;
        res=0.;
    }
    return res;
}

double v0(double t,double x,DataFile *df)
{
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double epsilon(df->getEpsilon());
    double xp;
    double c=1.;
    double sigma = 1.;
    double res;
    double tau;
    xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
    if (cas==4||cas==5||cas==6||cas==7||cas==8){
        //res = -epsilon*xp*exp(-0.5*pow(xp,2))/rho0(t,x,df);
        res = pow(xp,2);
        res = 0.5*res;
        res = -xp*exp(-res)*epsilon/rho0(t,x,df);
    }
    else if (cas==9){
        res = 0.;
    }
    else if (cas==10){
        tau = 2.*M_PI;
        res = epsilon*tau*cos(tau*x)/rho0(t,x,df);
    }
    else{
        //std::cout << "ce cas n'est pas défini" << std::endl;
        res=0.;
    }
    return res;
}

double w0(double t,double x,DataFile *df)
{
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double epsilon(df->getEpsilon());
    double nu(df->getNu());
    double kappa(df->getKappa());
    double xp;
    double c=1.;
    double sigma = 1.;
    double res;
    xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
    if (cas==4||cas==5){
        res = 1. + 2*kappa*nu/epsilon*v0(t,x,df);
    }
    else if (cas==6||cas==7||cas==8){
        res = cos(0.1*M_PI*x)+(2*kappa*nu/epsilon)*v0(t,x,df);
        //res = 1.;
    }
    else if (cas==9){
        res = 0. + 2*kappa*nu/epsilon*v0(t,x,df);
    }
    else if (cas==10){
        double tau = 2.*M_PI;
        res = cos(tau*x);
        //res = 0.;
    }
    else{
        //std::cout << "ce cas n'est pas défini" << std::endl;
        res=0.;
    }
    return res;
}

double intRho0(double t, double x, DataFile *df)
{
    //calcul de Rho0-C(x) avec C(x) = 1 pour tout x
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double xp;
    double c=1.;
    double sigma = 1.;
    double res;
    xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
    if (cas==8){
        //res = -sqrt(M_PI/2.)*erf(xp/sqrt(2.));
        //res = 2.*x;
        //res=0.;
        res = -sin(x);
    }
    else if (cas==10){
        double tau = 2.*M_PI;
        res = -cos(tau*x)/tau;
    }
    else {
        res = 0.;
    }
    return res;
}

double Psi0(double t, double x, DataFile *df)
{
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double xp;
    double c=1.;
    double sigma = 1.;
    double res;
    double lambda(df->getLambda());
    xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
    if (cas==8){
        //res = -(sqrt(0.5*M_PI)*xp*erf(xp/sqrt(2))+exp(-0.5*pow(xp,2)));
        //res = 1.+pow(x,2);
        //res = 0.;
        res = 1+cos(x);
    }
    else if (cas==10){
        double tau = 2.*M_PI;
        res = sin(tau*x)/pow(lambda*tau,2);
    }
    else {
        res = 0.;
    }
    return res;
}

double CdX(double x, DataFile *df)
{
    int cas(df->getCas());
    double CdX;
    if (cas==8){
        CdX = 1.;
    }
    else if (cas==9){
        if (1./3.<x<2./3.){
            CdX = 0.02;
        }
        else {
            CdX = 1.;
        }
    }
    else if (cas==10){
        CdX = 2.;
    }
    else {
        CdX = 0.;
    }
    return CdX;
}

void saveSol(DataFile *df, Eigen::VectorXd &vecU, int pdt)
{
    // n'est plus utilisé
    int cas(df->getCas());
    int nbPts(df->getNbPts());
    int Nx(df->getNx());
    std::string resultatChemin(df->getResultatChemin());
    std::string filename;
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    filename = resultatChemin+ "sol" + std::to_string(pdt) + ".dat";
    std::ofstream sol_file(filename);
    double x;
    if (nbPts == -1){
        for (int i=0; i<Nx; i++){
        x = xmin+0.5*dx+i*dx;
        sol_file << x << " " << vecU[i] << " " << vecU[i+Nx]/vecU[i] << " " << vecU[i+2*Nx]/vecU[i] << std::endl;
        }
    }
    else{
        int rapport = Nx/nbPts;
        int j;
        for(int i=0; i<nbPts; i++){
            j = i*rapport;
            x = xmin + 0.5*dx + j*dx;
            sol_file << x << " " << vecU[j] << " " << vecU[j+Nx]/vecU[j] << " " << vecU[j+2*Nx]/vecU[j] << std::endl;
        }
        sol_file << xmax-0.5*dx << " " << vecU[Nx-1] << " " << vecU[2*Nx-1]/vecU[Nx-1] << " " << vecU[3*Nx-1]/vecU[Nx-1] << std::endl;
    }
}

void saveEnergie(DataFile *df, std::vector<double> energieVec, std::vector<double> tempsVec){
    std::string resultatChemin(df->getResultatChemin());
    std::string filename;
    filename = resultatChemin+ "energie.dat";
    std::ofstream sol_file(filename);
    int nbImg(df->getNbImg());
    for (int i=0; i<nbImg-1; i++){
        sol_file << std::setprecision(10)<< tempsVec[i] << " " << energieVec[i] << std::endl;
    }
}

void FluxRusanov(Eigen::VectorXd &Flux, DataFile *df, Eigen::VectorXd &vecU, double t)
{
    int Nx(df->getNx());
    int cas(df->getCas());
    double xmin(df->getXMin());
    double xmax(df->getXMax());
    double dx = (xmax-xmin)/Nx;
    double gamma(df->getGamma());
    double r(df->getR());
    double kappa(df->getKappa());
    double nu(df->getNu());
    Eigen::VectorXd rho(Nx);
    Eigen::VectorXd Frho(Nx);
    Eigen::VectorXd u(Nx);
    Eigen::VectorXd Fu(Nx);
    Eigen::VectorXd v(Nx);
    Eigen::VectorXd Fv(Nx);
    Eigen::VectorXd maxLambda(Nx);
    Eigen::VectorXd lRusanov(Nx);
    for (int i=0; i<Nx; i++){
        rho[i] = vecU[i];
        u[i] = vecU[i+Nx]/vecU[i];
        v[i] = vecU[i+2*Nx]/vecU[i];
    }
    if(cas==0 || cas==1)
    {
        //pression en p^2/2
        for (int i=0; i<Nx; i++){
            Frho[i] = vecU[i+Nx];
            Fu[i] = u[i] * vecU[i+Nx] + 0.5*pow(vecU[i],2);
            Fv[i] = u[i] * vecU[i+2*Nx];
            maxLambda[i] = abs(u[i])+sqrt(rho[i]);
        }
    }
    else 
    {
        //pression en rho gamma
        for (int i=0; i<Nx; i++){
            Frho[i] = vecU[i+Nx];
            Fu[i] = u[i] * vecU[i+Nx] + pow(vecU[i],df->getGamma())+2.*r*kappa*nu*vecU[i];
            Fv[i] = u[i] * vecU[i+2*Nx];
            maxLambda[i] = abs(u[i])+sqrt(gamma*pow(rho[i],gamma-1)+2.*r*kappa*nu);
        }
    }
    for (int i=0; i<Nx-1; i++){
        lRusanov[i] = (maxLambda[i+1]<maxLambda[i]) ? maxLambda[i] : maxLambda[i+1];
    }
    if (cas==1){
        lRusanov[Nx-1] = (maxLambda[0]<maxLambda[Nx-1]) ? maxLambda[Nx-1] : maxLambda[0];
    }
    else {
        lRusanov[Nx-1] = (maxLambda[0]<maxLambda[Nx-1]) ? maxLambda[Nx-1] : maxLambda[0];
    }
    for (int i=0; i<Nx-1; i++){
        Flux[i] = 0.5*(Frho[i+1]+Frho[i])-0.5*lRusanov[i]*(vecU[i+1]-vecU[i]);
        Flux[i+Nx] = 0.5*(Fu[i+1]+Fu[i])-0.5*lRusanov[i]*(vecU[Nx+i+1]-vecU[Nx+i]);
        Flux[i+2*Nx] = 0.5*(Fv[i+1]+Fv[i])-0.5*lRusanov[i]*(vecU[2*Nx+i+1]-vecU[2*Nx+i]);
    }
    if (cas==1){
        Flux[Nx-1]=0.;
        Flux[2*Nx-1]=0.5;
        Flux[3*Nx-1]=0.;
    }
    else if (cas==2){
        //Dirichlet 
    }
    else if (cas==3){
        //conditions de Dirichlets pour le cas 3
        double C0 = 1.;
        double C1 = 1.;
    }
    else if (cas==4){
        //cas test construit dirichlet partout ?
        double rho = rho0(t,xmax,df);
        double w = w0(t,xmax,df);
        double v = v0(t,xmax,df);
        Flux[Nx-1]=rho*w;
        Flux[2*Nx-1]=rho*pow(w,2)+pow(rho,gamma)+2*kappa*nu*r*rho;
        Flux[3*Nx-1]=rho*w*v;
        //cas test avec dirichlet en rho et en v, neumann en u 
    }
    else if (cas==7||cas==8||cas==9){
        //Cas 6 -> condition de Dirichlet en rho et v, neumann en u
        //Cas 9 -> Jungel pas le cas 6 mais même CL
        double rho = rho0(t,xmax,df);
        double w = w0(t,xmax,df);
        double v = v0(t,xmax,df);
        double ub = vecU[2*Nx-1]/vecU[Nx-1]-0.5*dx*0;
        Flux[Nx-1] = rho*w;
        Flux[2*Nx-1] = rho*pow(ub,2)+pow(rho,gamma)+2*kappa*nu*r*rho;
        Flux[3*Nx-1] = rho*w*v;
    }
    else {
        //conditions periodiques
        Flux[Nx-1] = 0.5*(Frho[0]+Frho[Nx-1]) -0.5*lRusanov[Nx-1]*(vecU[0]-vecU[Nx-1]);
        Flux[2*Nx-1] = 0.5*(Fu[0]+Fu[Nx-1])-0.5*lRusanov[Nx-1]*(vecU[Nx]-vecU[2*Nx-1]);
        Flux[3*Nx-1] = 0.5*(Fv[0]+Fv[Nx-1]) -0.5*lRusanov[Nx-1]*(vecU[2*Nx]-vecU[3*Nx-1]);
    }
}

void FluxRusanov2(Eigen::VectorXd &Flux, DataFile *df, Eigen::VectorXd &vecU, double t)
{
    int Nx(df->getNx());
    int cas(df->getCas());
    double xmin(df->getXMin());
    double xmax(df->getXMax());
    double dx = (xmax-xmin)/Nx;
    double gamma(df->getGamma());
    double r(df->getR());
    double kappa(df->getKappa());
    double nu(df->getNu());
    Eigen::VectorXd rho(Nx);
    Eigen::VectorXd Frho(Nx);
    Eigen::VectorXd u(Nx);
    Eigen::VectorXd Fu(Nx);
    Eigen::VectorXd v(Nx);
    Eigen::VectorXd Fv(Nx);
    Eigen::VectorXd maxLambda(Nx);
    Eigen::VectorXd lRusanov(Nx);
    for (int i=0; i<Nx; i++){
        rho[i] = vecU[i];
        u[i] = vecU[i+Nx]/vecU[i];
        v[i] = vecU[i+2*Nx]/vecU[i];
    }
    //définition de la pression
    if (cas==0 || cas==1){
        //pression en p^2/2
        for (int i=0; i<Nx; i++){
            Frho[i] = vecU[i+Nx];
            Fu[i] = u[i] * vecU[i+Nx] + 0.5*pow(vecU[i],2);
            Fv[i] = u[i] * vecU[i+2*Nx];
            maxLambda[i] = abs(u[i])+sqrt(rho[i]);
        }
    }
    else {
        //pression en rho gamma
        for (int i=0; i<Nx; i++){
            Frho[i] = vecU[i+Nx];
            Fu[i] = u[i] * vecU[i+Nx] + pow(vecU[i],df->getGamma())+2.*r*kappa*nu*vecU[i];
            Fv[i] = u[i] * vecU[i+2*Nx];
            maxLambda[i] = abs(u[i])+sqrt(gamma*pow(rho[i],gamma-1)+2.*r*kappa*nu);
        }
    }
    for (int i=0; i<Nx-1; i++){
        lRusanov[i] = (maxLambda[i+1]<maxLambda[i]) ? maxLambda[i] : maxLambda[i+1];
    }
    if (cas==1){
        //cas dirichlet ? pas besoin de le définir
        lRusanov[Nx-1] = -1.;
    }
    else {
        //cas périodique
        lRusanov[Nx-1] = (maxLambda[0]<maxLambda[Nx-1]) ? maxLambda[Nx-1] : maxLambda[0];
    }
    //on défini les flux
    for (int i=0; i<Nx-1; i++){
        Flux[i] = 0.5*(Frho[i+1]+Frho[i])-0.5*lRusanov[i]*(vecU[i+1]-vecU[i]);
        Flux[i+Nx] = 0.5*(Fu[i+1]+Fu[i])-0.5*lRusanov[i]*(vecU[Nx+i+1]-vecU[Nx+i]);
        Flux[i+2*Nx] = 0.5*(Fv[i+1]+Fv[i])-0.5*lRusanov[i]*(vecU[2*Nx+i+1]-vecU[2*Nx+i]);
    }
    //flux aux bords
    if (cas==1){
        Flux[Nx-1]=0.;
        Flux[2*Nx-1]=0.5;
        Flux[3*Nx-1]=0.;
    }
    else if (cas==4){
        //cas test avec dirichlet en rho et en v, neumann en u 
        double rho = rho0(t,xmax,df);
        double w = w0(t,xmax,df);
        double v = v0(t,xmax,df);
        Flux[Nx-1]=rho*w;
        Flux[2*Nx-1]=rho*pow(w,2)+pow(rho,gamma)+2*kappa*nu*r*rho;
        Flux[3*Nx-1]=rho*w*v;
    }
    else if (cas==7||cas==8||cas==9){
        //Cas 6-9 -> condition de Dirichlet en rho et v, neumann en u
        double rho = rho0(t,xmax,df);
        double w = w0(t,xmax,df);
        double v = v0(t,xmax,df);
        double ub = vecU[2*Nx-1]/vecU[Nx-1]-0.5*dx*0;
        //ub = w;
        #if 0
        Flux[Nx-1] = rho*w;
        Flux[2*Nx-1] = rho*pow(w,2)+pow(rho,gamma)+2*kappa*nu*r*rho;
        Flux[3*Nx-1] = rho*w*v;
        #else 
        Flux[Nx-1] = rho*ub;
        Flux[2*Nx-1] = rho*pow(ub,2)+pow(rho,gamma)+2*kappa*nu*r*rho;
        Flux[3*Nx-1] = rho*ub*v;
        #endif
    }
    else {
        //conditions periodiques
        Flux[Nx-1] = 0.5*(Frho[0]+Frho[Nx-1]) -0.5*lRusanov[Nx-1]*(vecU[0]-vecU[Nx-1]);
        Flux[2*Nx-1] = 0.5*(Fu[0]+Fu[Nx-1])-0.5*lRusanov[Nx-1]*(vecU[Nx]-vecU[2*Nx-1]);
        Flux[3*Nx-1] = 0.5*(Fv[0]+Fv[Nx-1]) -0.5*lRusanov[Nx-1]*(vecU[2*Nx]-vecU[3*Nx-1]);
    }
}

void updateRusanov(Eigen::VectorXd &vecU, Eigen::VectorXd &Flux, DataFile *df, double dt, double t)
{
    int cas(df->getCas());
    double epsilon(df->getEpsilon());
    int Nx(df->getNx());
    double kappa(df->getKappa());
    double nu(df->getNu());
    double r(df->getR());
    double gamma(df->getGamma());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    if (cas==1){
        //adapte a des condtions de Dirichlet
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-0.);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-2.);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-0.);
    }
    else if (cas==2){
        //Dirichelt
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
        double rho0 = 2.+sin(0.2*M_PI*xmin);
        double rhou0 = 1.*2.+sin(0.2*M_PI*xmin);
        double rhov0 = 0.2*M_PI*epsilon*cos(0.2*M_PI*xmin);
        //vecU[0] = vecU[0] - dt/dx*(Flux[0]-rho0);
        //vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-rhou0);
        //vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-rhov0);
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-0.);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-0.);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-0.);
    }
    else if (cas==3){
        //Dirichelt
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
        double rho0 = 1./sqrt(M_PI)*exp(-pow(xmin,2));
        double rhou0 = 1.*1./sqrt(M_PI)*exp(-pow(xmin,2));
        double rhov0 = -epsilon*2*xmin/M_PI*exp(-2.*pow(xmin,2));
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-rho0);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-rhou0);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-rhov0);
    }
    else if (cas==4){
        //Dirichlet
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
        double rhov = rho0(t,xmax,df);
        double wv = w0(t,xmin,df);
        double vv = v0(t,xmin,df);
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-rhov*wv);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-(rhov*pow(wv,2)+pow(rhov,gamma)+2*kappa*nu*r*rhov));
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-rhov*wv*vv);
    }
    else if (cas==7||cas==8||cas==9){
        //cas 7 et 8 sont le meme que cas 6 avec CL Dirichlet en rho et v et neumann en u
        //cas 9 meme que jungel avec mêmes CL
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
        //cas dirichlet pour rho et v, neumann pour u;
        double rhov = rho0(t,xmin,df);
        double wv = w0(t,xmin,df);
        double vv = v0(t,xmin,df);
        double F0,F1,F2;
        F0 = rhov*wv;
        F1 = rhov*pow(vecU[0]-0.5*dx*0.,2)+pow(rhov,gamma)+2*kappa*nu*r*rhov;
        F2 = rhov*wv*vv;
        //vecU[0] = vecU[0] -dt/dx*(F0-Flux[Nx-1]);
        //vecU[Nx] = vecU[Nx] -dt/dx*(F1-Flux[2*Nx-1]);
        //vecU[2*Nx] = vecU[2*Nx] -dt/dx*(F2-Flux[3*Nx-1]);
        vecU[0] = vecU[0] -dt/dx*(Flux[0]-F0);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-F1);
        //vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-Flux[2*Nx-1]);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-F2);
    }
    else {
        //CL periodique
        vecU[0] = vecU[0] -dt/dx*(Flux[0]-Flux[Nx-1]);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-Flux[2*Nx-1]);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-Flux[3*Nx-1]);
        for (int i=1; i<Nx; i++){
            vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
            vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
            vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
        }
    }
}

void updateRusanov2(Eigen::VectorXd &vecU, Eigen::VectorXd &Flux, DataFile *df, double dt, double t)
{
    int cas(df->getCas());
    double epsilon(df->getEpsilon());
    int Nx(df->getNx());
    double kappa(df->getKappa());
    double nu(df->getNu());
    double r(df->getR());
    double gamma(df->getGamma());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    //conditions limites si besoin
    if (cas==1){
        //dirichlet pour le cas specifique
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-0.);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-2.);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-0.);
    }
    else if (cas==2){
        //dirichlet particulier
        double rho0 = 2.+sin(0.2*M_PI*xmin);
        double rhou0 = 1.*2.+sin(0.2*M_PI*xmin);
        double rhov0 = 0.2*M_PI*epsilon*cos(0.2*M_PI*xmin);
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-0.);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-0.);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-0.);
    }
    else if (cas==3){
        //dirichlet particulier
        double rho0 = 1./sqrt(M_PI)*exp(-pow(xmin,2));
        double rhou0 = 1.*1./sqrt(M_PI)*exp(-pow(xmin,2));
        double rhov0 = -epsilon*2*xmin/M_PI*exp(-2.*pow(xmin,2));
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-rho0);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-rhou0);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-rhov0);
    }
    else if (cas==4){
        //dirichlet generale
        double rhov = rho0(t,xmax,df);
        double wv = w0(t,xmin,df);
        double vv = v0(t,xmin,df);
        vecU[0] = vecU[0] - dt/dx*(Flux[0]-rhov*wv);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-(rhov*pow(wv,2)+pow(rhov,gamma)+2*kappa*nu*r*rhov));
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-rhov*wv*vv);
    }
    else if (cas==7 || cas==8 || cas==9){
        double rhov = rho0(t,xmin,df);
        double wv = w0(t,xmin,df);
        double vv = v0(t,xmin,df);
        double F0,F1,F2;
        double ub = vecU[Nx]/vecU[0]-0.5*dx*0.;
        //ub = wv;
        #if 0
        F0 = rhov*wv;
        F1 = rhov*pow(wv,2)+pow(rhov,gamma)+2*kappa*nu*r*rhov;
        F2 = rhov*wv*vv;
        #else
        F0 = rhov*ub;
        F1 = rhov*pow(ub,2)+pow(rhov,gamma)+2*kappa*nu*r*rhov;
        F2 = rhov*ub*vv;
        #endif
        vecU[0] = vecU[0] -dt/dx*(Flux[0]-F0);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-F1);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-F2);
    }
    else {
        //CL periodique
        vecU[0] = vecU[0] -dt/dx*(Flux[0]-Flux[Nx-1]);
        vecU[Nx] = vecU[Nx] -dt/dx*(Flux[Nx]-Flux[2*Nx-1]);
        vecU[2*Nx] = vecU[2*Nx] -dt/dx*(Flux[2*Nx]-Flux[3*Nx-1]);
    }
    //on complète pour le reste des valeurs 
    for (int i=1; i<Nx; i++){
        vecU[i] = vecU[i] -dt/dx*(Flux[i]-Flux[i-1]);
        vecU[i+Nx] = vecU[i+Nx] -dt/dx*(Flux[Nx+i]-Flux[Nx+i-1]);
        vecU[i+2*Nx] = vecU[i+2*Nx] -dt/dx*(Flux[2*Nx+i]-Flux[2*Nx+i-1]);
    }
}

double calculEnergie(Eigen::VectorXd &vecU, DataFile *df)
{
    double energie = 0.;
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    double epsilon(df->getEpsilon());
    double gamma(df->getGamma());
    double r(df->getR());
    double tempo;
    double tmp1;
    double tmp2;
    double cst = 1. + 4.*kappa*(1.-kappa)*nu*nu/(epsilon*epsilon);
    for (int i=0; i<Nx; i++){
        //tempo = 0.5*(vecU[i+Nx]*vecU[i+Nx]/vecU[i] + (4.*kappa*nu*nu*(1.-kappa)/epsilon)*vecU[i+2*Nx]*vecU[i+2*Nx]/vecU[i])+2.*r*kappa*nu*vecU[i]*log(vecU[i]);
        tmp1 = 0.5*(vecU[i+Nx]*vecU[i+Nx]/vecU[i] + cst*vecU[i+2*Nx]*vecU[i+2*Nx]/vecU[i]) + 2*r*kappa*nu*vecU[i]*log(vecU[i]);
        if (cas==0 || cas==1){
            tmp2 = vecU[i]*vecU[i]/2.;
        }
        else {
            tmp2 =  pow(vecU[i], gamma)/(gamma-1);
        }
        energie = energie + tmp1 + tmp2;
    }
    return energie;
}

void BuildSource(Eigen::VectorXd &source, Eigen::VectorXd &rho, DataFile *df, double dt, double temps)
{
    // n'est plus utilisé
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double r(df->getR());
    double kappa(df->getKappa());
    double nu(df->getNu());
    double cst;
    Eigen::VectorXd tempo(Nx);
    cst = 4*dt*r*kappa*nu/dx;
    if (abs(rho[0]-rho[Nx-1])<10e-13) {
        tempo[0] = rho[0];
    }
    else {
        tempo[0] = (rho[0]-rho[Nx-1])/(log(rho[0])-log(rho[Nx-1]));
    }
    for (int i=1; i<Nx; i++){
        if (abs(rho[i]-rho[i-1])<10e-13) {
            tempo[i] = rho[i];
        }
        else {
            tempo[i] = (rho[i]-rho[i-1])/(log(rho[i])-log(rho[i-1]));
        }
    }
    for (int i=0; i<Nx; i++){
        source[i] = 0.;
    }
    source[Nx] = cst*(tempo[0]-tempo[Nx-1])/(sqrt(tempo[0]*rho[0]));
    for (int i=1; i< Nx; i++){
        source[Nx+1] = cst*(tempo[i]-tempo[i-1])/(sqrt(tempo[i]*rho[i]));
    }
}

Eigen::SparseMatrix<double> LaplacianPsi(DataFile *df)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double lambda=1.;
    Eigen::SparseMatrix<double> Mat(Nx,Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = pow(lambda,2)/pow(dx,2);
    double val0 = 2.*cst;
    double valb = 3.*cst;
    double valx = -cst;
    
    //on construit la matrice avec des conditions periodique de base (ne fonctionne pas)
    if (cas==7|cas==8||cas==9){
        //conditions de Dirichlet au bord
        tL.push_back(T(0,0,valb));
        tL.push_back(T(0,1,valx));
    }
    else {
        tL.push_back(T(0,0,val0));
        tL.push_back(T(0,1,valx));
        tL.push_back(T(0,Nx-1,valx));
    }
    //lignes
    for (int i=1; i<Nx-1; i++){
        tL.push_back(T(i,i-1,valx));
        tL.push_back(T(i,i,val0));
        tL.push_back(T(i,i+1,valx));
    }
    //ligne N
    if (cas==7||cas==8||cas==9){
        tL.push_back(T(Nx-1,Nx-2,valx));
        tL.push_back(T(Nx-1,Nx-1,valb));
    }
    else {
        tL.push_back(T(Nx-1,0,valx));
        tL.push_back(T(Nx-1,Nx-2,valx));
        tL.push_back(T(Nx-1,Nx-1,val0));
    }
    Mat.setFromTriplets(tL.begin(),tL.end());
    //std::cout << "j affiche mat psi" << std::endl;
    //std::cout << std::setprecision(6) << MatrixXd(Mat) << std::endl;
    return Mat; 
}

void DefDoping(Eigen::VectorXd &vecC, DataFile *df, double t)
{
    // n'est plus utilisé
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double x;
    if (cas==0){
        for (int i=0; i<Nx; i++){
            vecC[i] = 0.5;
        }
    }
    else if (cas==4||cas==5){
        //gere le terme source directement dans C(t,x)
        for (int i=0; i<Nx; i++){
            x = xmin + 0.5*dx + i*dx;
            vecC[i] = 1.-rho0(t,x,df);
        }
    }
    else {
        double x;
        for (int i=0; i<Nx; i++){
            x = 0.5*dx + i*dx + xmin;
            vecC[i] = pow(x,2);
        }
    }
}

void PoissonSM(Eigen::VectorXd &PSM, Eigen::VectorXd &rho, Eigen::VectorXd &vecC, DataFile *df)
{
    //n'est pas utilisé pour l'instant.
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin()); 
    for (int i=0; i<Nx; i++){
        //on traite lambda apres
        PSM[i] = (rho[i] - vecC[i]);
    }
}

void BuildV(Eigen::VectorXd &vecV, double t, Eigen::VectorXd &rho, Eigen::SparseMatrix<double> DC, int incr, DataFile *df)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    double epsilon(df->getEpsilon());
    double lambda(df->getLambda());
    Eigen::SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solverDb;
    Eigen::VectorXd PsmTempo(Nx);
    double x;
    double xp;
    double c=1.;
    double cstl;
    for (int i=0; i<Nx; i++){
        x = xmin+i*dx+0.5*dx;
        PsmTempo[i] = (1./pow(lambda,2))*(rho[i]-CdX(x,df));
        //PsmTempo[i] = rho0(t,x,df)-1.;
    }
    solverDb.compute(DC);
    if (cas==7||cas==8){
        cstl = 2./pow(dx,2);
        //On ajoute les conditions de Dirichlet avec le solution exacte
        //xp = xmin-c*t+(xmax-xmin)*(1.+floor((c*t-xmin+xmin)/(xmax-xmin)));
        PsmTempo[0] += cstl*Psi0(t,xmin,df);
        //xp = xmax-c*t+(xmax-xmin)*(1.+floor((c*t-xmax+xmin)/(xmax-xmin)));
        PsmTempo[Nx-1] += cstl*Psi0(t,xmax,df);
    }  
    else if (cas==9){
        //Conditions de Dirichlet sans connaître la solution exacte.
        cstl = 2./pow(dx,2);
        //on regarde pour incrementer la tension au cours du temps
        double cd = 0.1*incr;
        #if 0
        if (t<50){
            cd = 0.25;
        }
        else {
            cd = 0.29;
        }
        #endif
        PsmTempo[0] += cstl*0.;
        //PsmTempo[Nx-1] += 2.*0.5/pow(dx,2);
        PsmTempo[Nx-1] = cd*cstl;
        }
    else if (cas==10){
        cstl = 2./pow(dx,2);
        PsmTempo[0] = cstl*Psi0(t,xmin,df);
        PsmTempo[Nx-1] = cstl*Psi0(t,xmax,df);
    }    
    //on rajoute un terme source test
    for (int i=0; i<Nx; i++){
        x = xmin+i*dx+0.5*dx;
        PsmTempo[i] += cos(x)+1.-rho0(t,x,df);
    }
    //std::cout << "le second membre est " << std::endl;
    //std::cout << PsmTempo << std::endl;
    vecV = solverDb.solve(PsmTempo);
    //std::cout << "vecV = " << vecV << std::endl;
}

Eigen::VectorXd BuildSource2(Eigen::VectorXd &rho, Eigen::VectorXd &vecV, DataFile *df, double dt, double t)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double nu(df->getNu());
    double kappa(df->getKappa());
    double gamma(df->getGamma());
    double epsilon(df->getEpsilon());
    double lambda(df->getLambda());
    double dx = (xmax-xmin)/Nx;
    double cst;
    double r(df->getR());
    Eigen::VectorXd tempo(Nx);
    Eigen::VectorXd source(2*Nx);
    //cst = 4*dt*r*kappa*nu/dx;
    //C'est un choix de discretisation par default, pour l'instant on se place dans le cas r=0
    //source[0] = dt*rho[0]*(vecV[1]-vecV[0])/dx;
    //source[Nx-1] = dt*rho[Nx-1]*(vecV[Nx-1]-vecV[Nx-2])/dx;
    //for(int i=1; i<Nx-1; i++){
    //    source[i] = dt*rho[i]*(vecV[i+1]-vecV[i-1])/(2.*dx);
    //}
    for (int i=0; i<2*Nx; i++){
        source[i] = 0.;
    }
    //on ajoute le cas du terme source
    if (cas==4||cas==5){
        double x;
        double xp;
        double c=1.;
        double rhov;
        double rc;
        double inter;
        double inter2;
        double val_period;
        double rhoval;
        double sigma =1.;
        double dxrho, dxxrho, dxxxrho, erc;
        for (int i=0; i<Nx; i++){
            #if 0
            x = xmin+0.5*dx+i*dx;
            xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
            rhov = rho0(t,x,df);
            rc = 0.5*pow(xp,2);
            inter2 = (-xp*exp(-rc)/rhov+xp*exp(-2.*rc)/pow(rhov,2))*(rc*exp(-rc)/rhov+2.-2.*rc);
            inter2 += (exp(-rc)/rhov)*((1.-rc)*xp*exp(-rc)/rhov + xp*rc*exp(-2.*rc)/pow(rhov,2) -2.*xp);
            inter = -gamma*pow(rhov,gamma-1)*xp*exp(-rc);
            inter += pow(epsilon,2)*rhov*inter2;
            //inter += rhov*2*x;
            //source[i] += inter+r*c*rhov;
            #else
            x = xmin+0.5*dx+i*dx;
            val_period = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
            rhoval = rho0(t,x,df); 
            rc = (pow(val_period,2))/(2.*sigma);
            inter2 = (-val_period*exp(-rc)/(sigma*rhoval)+val_period/sigma*(exp(-2.*rc)/pow(rhoval,2)))*(rc*exp(-rc)/rhoval+2.-2.*rc);
            inter2 += exp(-rc)/rhoval*((1-rc)*val_period/(sigma*rhoval)*exp(-rc)+val_period/(sigma*pow(rhoval,2))*rc*exp(-2.*rc)-2.*val_period/sigma);
            inter = -gamma*pow(rhoval,gamma-1)*val_period*exp(-rc)/sigma;
            inter += 1.*pow(epsilon,2)*rhoval/sigma*inter2;
            //source[i] = (inter)+r*c*rhoval;
            #endif
            
            x = xmin+0.5*dx+i*dx;
            xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
            rhov = rho0(t,x,df);
            erc = exp(-0.5*pow(xp,2));
            dxrho = -xp*erc;
            dxxrho = (pow(xp,2)-1.)*erc;
            dxxxrho = xp*erc*(3-pow(xp,2));
            inter2 = gamma*pow(rhov,gamma-1)*dxrho;
            inter = dxxxrho + pow(dxrho,3)/(pow(rhov,2)) - 2*dxrho*dxxrho/rhov;
            source[i] = -pow(epsilon,2)*inter+inter2;
            
        }
    }
    else if (cas==6 || cas==7||cas==8){
        double x, xp;
        double c=1.;
        double sigma=1.;
        double pi=0.1*M_PI; //idee brillante
        double rc, erc; 
        double rhov, dxrho, dxxrho, dxxxrho, dtrho, dtxrho;
        double u, dxu, dxxu;
        double w, dxw, dxxw, dtw;
        double v, dtv, dxv;
        double dtrhow, dxrhoww, dxwdxrho, dxrhodxw;
        double A,B,C,D,bohm, dxp;
        double dtrhou, dxrhouxu, dxrhodxu, ter;
        for (int i=0; i<Nx; i++){
            x = xmin+i*dx+0.5*dx;
            xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
            //xp = x;
            rc = 0.5*pow(xp,2);
            erc = exp(-rc);
            rhov = rho0(t,x,df);
            v = v0(t,x,df);
            dtrho = c*xp*erc;
            dxrho = -xp*erc;
            dxxrho = (pow(xp,2)-1.)*erc;
            dxxxrho = xp*erc*(3-pow(xp,2));
            dtxrho = c*erc*(sigma-pow(xp,2))/pow(sigma,2);
            #if 1
            u = cos(pi*x);
            dxu = -pi*sin(pi*x);
            dxxu = -pow(pi,2)*cos(pi*x);
            #else 
            u=1.;
            dxu = 0.;
            dxxu = 0.;
            #endif
            w = w0(t,x,df);
            dxw = dxu + 2*kappa*nu*(dxxrho/rhov - pow(dxrho,2)/pow(rhov,2));
            //dxxw = dxxu + 2*kappa*nu*(dxxrho/rhov - pow(dxrho,2)/pow(rhov,2));
            dxxw = dxxu + 2*kappa*nu*((dxxxrho/rhov - dxrho*dxxrho/pow(rhov,2))-2*(dxrho/rhov)*(dxxrho/rhov - pow(dxrho/rhov,2)));
            dtw = 2*kappa*nu*(dtxrho/rhov - dtrho*dxrho/pow(rhov,2));

            //deuxieme composante
            dtv = epsilon*(dtxrho/rhov - dxrho*dtrho/(pow(rhov,2)));
            dxv = epsilon*(dxxrho/rhov - pow(dxrho,2)/pow(rhov,2));
            //B = rhov*dtv + v*dtrho + rhov*dxv + v*(u*dxrho+rhov*dxu) + epsilon*(dxrho*dxu + rhov*dxxu);
            B = rhov*dtv + v*dtrho + rhov*u*dxv + rhov*v*dxu + u*v*dxrho + epsilon*(dxrho*dxu + rhov*dxxu);
            //B = rhov*dtv + v*dtrho + rhov*u*dxv + rhov*v*dxu + u*v*dxrho;
            //B = rhov*u*dxv + rhov*v*dxu + u*v*dxrho;
            source[i+Nx] = B;

            //premiere composante 
            dxp = gamma*pow(rhov,gamma-1)*dxrho;
            bohm = dxxxrho + pow(dxrho,3)/(pow(rhov,2)) - 2*dxrho*dxxrho/rhov;
            dtrhou = u*dtrho;
            dxrhouxu = 2*rhov*u*dxu + pow(u,2)*dxrho;
            dxrhodxu = rhov*dxxu +dxrho*dxu;
            ter = r*rhov*u;
            A = -pow(epsilon,2)*bohm + dxp + dtrhou + dxrhouxu -2*nu*dxrhodxu + ter; 
            //A = dxp + dtrhou + dxrhouxu;
            //A = dxp + dxrhouxu;
            //std::cout << "A,i = " << A << " " << i << std::endl;
            if (cas==8){
                //on rajoute la partie en rho grad Psi
                C = rhov*intRho0(t,x,df);
                //C = 0.;
            }
            else {
                C = 0.;
            }
            source[i] = A+2*kappa*nu/epsilon*B-C;
            //source[i] = A;
            //on retente la premiere composante
            dtrhow = rhov*dtw + w*dtrho;
            dxrhoww = pow(w,2)*dxrho + 2*rhov*w*dxw;
            dxwdxrho = dxw*dxrho + w*dxxrho;
            dxrhodxw = dxw*dxrho + rhov*dxxw;
            //A = dtrhow + dxrhoww + dxp + ter -(pow(epsilon,2)-4*kappa*(1-kappa)*pow(nu,2))*bohm -2*kappa*nu*dxwdxrho -2*(1-kappa)*nu*dxrhodxw;
            //A = dtrhow + dxrhoww + dxp + ter -pow(epsilon,2)*bohm -2*kappa*nu*dxwdxrho-2*nu*(1-kappa)*dxrhodxu;
            //source[i] = A;
        }
    }
    else if (cas==10){
        //cas test 10
        double x;
        double tau = 2.*M_PI;
        double rho, u, v;
        double dtrho, dxrho, dxxrho, dxxxrho, dtxrho;
        double dxu, dxxu;
        double dtv, dxv; 
        double dtrhou, dtrhov, dxrhouxu, dxrhouxv, dxrhodxu, bohm, dxp, ter;
        double A,B,C;
        for (int i=0; i<Nx; i++){
            x = xmin + 0.5*dx + i*dx;
            rho = rho0(t,x,df);
            u = w0(t,x,df);
            v = v0(t,x,df);
            dxrho = tau*cos(tau*x);
            dxxrho = -pow(tau,2)*sin(tau*x);
            dxu = -tau*sin(tau*x);
            //dxu = 0.;
            dxxu = -pow(tau,2)*cos(tau*x);
            //dxxu = 0.;
            //deuxieme composante
            dtrhov = 0.;
            dxrhouxv = epsilon*(dxxrho*u+dxrho*dxu);
            dxrhodxu = dxrho*dxu+rho*dxxu;
            B = dtrhov + dxrhouxv + epsilon*dxrhodxu;
            source[i+Nx] = B;

            //premiere composante
            dxp = gamma*pow(rho,gamma-1)*dxrho;
            dxxxrho = pow(tau,3)*sin(tau*x);
            bohm = dxxxrho + pow(dxrho,3)/pow(rho,2) - 2.*dxrho*dxxrho/rho;
            ter = r*rho*u;
            dtrhou = 0.;
            dxrhouxu = 2.*rho*u*dxu + pow(u,2)*dxrho;
            A = dtrhou + dxrhouxu + dxp -pow(epsilon,2)*bohm -2.*nu*dxrhodxu +ter;
            C = rho*intRho0(t,x,df)/pow(lambda,2);
            C = 0.;
            source[i] = A+2.*kappa*nu/epsilon*B+C;

        }
    }

    //terme source sur la matrice
    if (cas==7||cas==8||cas==9){
        //On ajoute aux termes sources ce qui dépend des conditions limites de Dirichlet et Neumann homogene

        //ajout du terme diagonale haut gauche
        source[0] += -2.*nu*rho0(t,xmin,df)*0./dx;
        source[Nx-1] += -2.*nu*rho0(t,xmax,df)*0./dx;
        
        //ajout du terme extra diagonale bas gauche
        //source[0] += epsilon*rho0(t,xmin,df)*0./dx;
        //source[Nx-1] += epsilon*rho0(t,xmax,df)*0./dx;
        source[0+Nx] += epsilon*rho0(t,xmin,df)*0./dx;
        source[Nx-1+Nx] += epsilon*rho0(t,xmax,df)*0./dx;

        //ajout du terme extra diagonale haut droit 
        //moralement ce terme qui approxime dx rho, doit valoir 0 au bord ?
        double thd = epsilon-4.*kappa*(1.-kappa)*pow(nu,2)/epsilon;
        //source[Nx] += 2.*rho0(t,xmin,df)*v0(t,xmin,df)/pow(dx,2);
        //source[2*Nx-1] += 2.*rho0(t,xmax,df)*v0(t,xmax,df)/pow(dx,2);
        //source[Nx] += thd*2.*rho0(t,xmin,df)*v0(t,xmin,df)/pow(dx,2);
        //source[2*Nx-1] += thd*2.*rho0(t,xmax,df)*v0(t,xmax,df)/pow(dx,2);
        
        //source[0] += thd*2.*rho0(t,xmin,df)*v0(t,xmin,df)/pow(dx,2);
        //source[Nx-1] += thd*2.*rho0(t,xmax,df)*v0(t,xmax,df)/pow(dx,2);

        //ajout du terme diagonale bas droit
        source[Nx] += 2.*kappa*nu*1.*rho0(t,xmin,df)*v0(t,xmin,df)/pow(dx,2);
        source[2*Nx-1] += 2.*kappa*nu*1.*rho0(t,xmax,df)*v0(t,xmax,df)/pow(dx,2);
    }
    if (cas==8||cas==9||cas==11){
        double x;
        //on ajoute le terme source en rho grad V
        source[0] += rho[0]*(vecV[1]-vecV[0])/dx;
        source[Nx-1] += rho[Nx-1]*(vecV[Nx-1]-vecV[Nx-2])/dx;
        for(int i=1; i<Nx-1; i++){
            source[i] += rho[i]*(vecV[i+1]-vecV[i-1])/(2.*dx);
        }
        //source[0] += rho0(t,xmin+0.5*dx,df)*intRho0(t,xmin,df)
        //source[0] += rho0(t,xmin,df)*((vecV[1]-vecV[0])/dx);
        //source[Nx-1] += rho0(t,xmax,df)*((vecV[Nx-1]-vecV[Nx-2])/dx);
        for (int i=1; i<Nx-1; i++){
            x = xmin+0.5*dx+i*dx;
            //source[i] += rho0(t,x,df)*(vecV[i+1]-vecV[i-1])/(2.*dx);
            //source[i] +=rho0(t,x,df)*2.;
        }
        //std::cout << "VecV dans la source = " << vecV << std::endl;
        if (vecV[0]>10e-5){
            //std::cout << "VecV n'est pas nul" << std::endl;
        }
    }

    #if 1
    //calcul du terme en r 4 nu kappa dx rho - periodique uniquement 
    cst = 4*r*kappa*nu/dx;
    if (abs(rho[0]-rho[Nx-1])<10e-13) {
        tempo[0] = rho[0];
    }
    else {
        tempo[0] = (rho[0]-rho[Nx-1])/(log(rho[0])-log(rho[Nx-1]));
    }
    for (int i=1; i<Nx; i++){
        if (abs(rho[i]-rho[i-1])<10e-13) {
            tempo[i] = rho[i];
        }
        else {
            tempo[i] = (rho[i]-rho[i-1])/(log(rho[i])-log(rho[i-1]));
        }
    }
    source[0] += cst*sqrt(rho[0]/tempo[0])*(rho[0]-rho[Nx-1]);
    for (int i=1; i< Nx; i++){
        source[i] += cst*sqrt(rho[i]/tempo[i])*(rho[i]-rho[i-1]);
    }
    //std::cout << "on affiche le terme source" << std::endl;
    //std::cout << source << std::endl;
    #endif
    return source;
}

Eigen::VectorXd BuildSourceRho(DataFile *df, double t)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double epsilon(df->getEpsilon());
    double nu(df->getNu());
    double kappa(df->getKappa());
    double c=1.;
    double sigma=1.;
    double x; 
    Eigen::VectorXd source(Nx);
    for (int i=0; i<Nx; i++){
        source[i] = 0.;
    }
    if (cas==6 || cas==7 || cas==8){
        //on ajoute un terme source afin de verifier la solution
        double rhov;
        double dxrhov, dtrhov;
        double dxxrhov;
        double u;
        double dxu;
        double rc;
        double erc;
        double pi=0.1*M_PI;
        double xp;
        double sigma=1.;
        double c=1.;
        for (int i=0; i<Nx; i++){
            x = xmin + 0.5*dx + i*dx;
            xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
            xp = x;
            rc = 0.5*pow(xp,2);
            erc = exp(-rc);
            rhov = rho0(t,x,df);
            #if 1
            u = cos(pi*x);
            dxu = -pi*sin(pi*x);
            #else
            u = 1.;
            dxu = 0.;
            #endif 
            dtrhov = c*xp*erc;
            dxrhov = -xp*erc;
            source[i] = dtrhov + u*dxrhov + rhov*dxu;
            //source[i] = u*dxrhov+rhov*dxu;
        }
    }
    else if (cas==10){
        double rho,u;
        double dxrho,dxu;
        double x;
        double tau = 2.*M_PI;
        double dtrho, dxrhou;
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rho = rho0(t,x,df);
            u = w0(t,x,df);
            dxrho = tau*cos(tau*x);
            dxu = -tau*sin(tau*x);
            //dxu = 0.;
            dtrho = 0.;
            dxrhou = rho*dxu+u*dxrho;
            source[i] = dtrho+dxrhou;
        }
    }
    if (cas==-1){
        //conditions de Dirichlet aux bords avec kappa specifique
        source[0] = 2*kappa*nu/pow(dx,2)*rho0(t,xmin,df);
        source[Nx-1] = 2*kappa*nu/pow(dx,2)*rho0(t,xmax,df);
    }
    return source;
}

void saveSol2(DataFile *df, Eigen::VectorXd &vecU, Eigen::VectorXd &vecV, int pdt)
{
    int cas(df->getCas());
    int nbPts(df->getNbPts());
    int Nx(df->getNx());
    std::string resultatChemin(df->getResultatChemin());
    std::string filename;
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double tmax(df->getTmax());
    double temps = tmax*pdt/2000.;
    filename = resultatChemin+ "sol" + std::to_string(pdt) + ".dat";
    std::ofstream sol_file(filename);
    double x;
    if (nbPts == -1){
        for (int i=0; i<Nx; i++){
        x = xmin+0.5*dx+i*dx;
        sol_file << x << " " << vecU[i] << " " << vecU[i+Nx]/vecU[i] << " " << vecU[i+2*Nx]/vecU[i] << " " << vecV[i] << " " << x/temps << " " << vecU[i+Nx] << " " << vecU[i+2*Nx] << std::endl;
        }
    }
    else{
        int rapport = Nx/nbPts;
        int j;
        for(int i=0; i<nbPts; i++){
            j = i*rapport;
            x = xmin + 0.5*dx + j*dx;
            sol_file << x << " " << vecU[j] << " " << vecU[j+Nx]/vecU[j] << " " << vecU[j+2*Nx]/vecU[j] << std::endl;
        }
        sol_file << xmax-0.5*dx << " " << vecU[Nx-1] << " " << vecU[2*Nx-1]/vecU[Nx-1] << " " << vecU[3*Nx-1]/vecU[Nx-1] << " " << x/temps << std::endl;
    }
    //for (int i=0; i<Nx; i++){
    //    x = xmin+0.5*dx+i*dx;
    //    sol_file << x << " " << vecU[i] << " " << vecU[i+Nx]/vecU[i] << " " << vecU[i+2*Nx]/vecU[i] << std::endl;
    //}
}

void saveExacte(DataFile *df, Eigen::VectorXd &vecU, double t)
{
    //Enregistre la solution donnée par les fonctions de définition a un temps donné, ne fonctionne que si on connaît déjà la solution
    // avec un terme source par exemple
    int cas(df->getCas());
    int nbPts(df->getNbPts());
    int Nx(df->getNx());
    double xmin(df->getXMin());
    double xmax(df->getXMax());
    double dx = (xmax-xmin)/Nx;
    double tmax(df->getTmax());
    double x;
    std::string chemin(df->getResultatChemin());
    std::string nomDeFichier;
    nomDeFichier = chemin + "sol_exacte.dat";
    std::ofstream sol_file(nomDeFichier);
    for(int i=0; i<Nx; i++){
        x = xmin+ 0.5*dx + i*dx;
        sol_file << x << " " << rho0(t,x,df) << " " << w0(t,x,df) <<  " " << v0(t,x,df) << " " << rho0(t,x,df)*w0(t,x,df) << " " << rho0(t,x,df)*v0(t,x,df)<< std::endl;
    }
    nomDeFichier = chemin + "diff_sol.dat";
    std::ofstream sol_file2(nomDeFichier);
    for(int i=0; i<Nx; i++){
        x = xmin+ 0.5*dx + i*dx;
        sol_file2 << x << " " << rho0(t,x,df) - vecU[i]<< " " << w0(t,x,df)-vecU[i+Nx]/vecU[i]<< " " << v0(t,x,df) - vecU[i+2*Nx]/vecU[i];
        sol_file2 << " " << rho0(t,x,df)*w0(t,x,df) -vecU[i+Nx]<< " " << rho0(t,x,df)*v0(t,x,df) -vecU[i+2*Nx]<< std::endl;
    }
}

void erreurL2(Eigen::VectorXd VecU, DataFile *df, double t)
{
    int Nx(df->getNx());
    int cas(df->getCas());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double x;
    double epsilon(df->getEpsilon());
    double nu(df->getNu());
    double c=1.;
    double sigma=1.;
    Eigen::VectorXd ErrVec(5*Nx);
    double errrho = 0.;
    double errw = 0.;
    double errv = 0.;
    double errrhow = 0.;
    double errrhov = 0.;
    double rhot, wt, vt;
    for (int i=0; i<5*Nx; i++){
        ErrVec[i] = 0.;
    }
    if (cas==4||cas==5||cas==6||cas==7||cas==8||cas==10){
        //on regarde l'erreur pour les cas ou on connaît la solution.
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            rhot = rho0(t,x,df);
            wt = w0(t,x,df);
            vt = v0(t,x,df);
            ErrVec[i] = rhot - VecU[i];
            ErrVec[i+Nx] = rhot*wt-VecU[i+Nx];
            ErrVec[i+2*Nx] = wt - VecU[i+Nx]/VecU[i];
            ErrVec[i+3*Nx] = rhot*vt-VecU[i+2*Nx];
            ErrVec[i+4*Nx] = vt - VecU[i+2*Nx]/VecU[i];
        }
    }
    for (int i=0; i<Nx; i++){
        errrho += pow(ErrVec[i],2)*dx;
        errw += pow(ErrVec[i+Nx],2)*dx;
        errrhow += pow(ErrVec[i+2*Nx],2)*dx;
        errv += pow(ErrVec[i+3*Nx],2)*dx;
        errrhov += pow(ErrVec[i+4*Nx],2)*dx;
    }
    std::string resultatChemin(df->getResultatChemin());
    std::string filename;
    filename = resultatChemin+"erreur.dat";
    std::ofstream sol_file(filename);
    sol_file << sqrt(errrho) << " " << sqrt(errw) << " " << sqrt(errv) << " " << sqrt(errrhow) << " " << sqrt(errrhov) << std::endl;
}

Eigen::SparseMatrix<double> TermeDzDrho(DataFile *df, double dt, Eigen::VectorXd(VecRho))
{
    //matrix D+E d rho dz 
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(Nx,Nx);
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(3*Nx);
    double tempo;
    double lambda=dt/pow(dx,2);
    double cst = 1.;
    //ligne 1 
    tempo = 2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0];
    tripletList.push_back(T(0,0,tempo));
    tempo = -(1.+VecRho[1]/VecRho[0]);
    tripletList.push_back(T(1,0,tempo));
    tempo = -(1.+VecRho[Nx-1]/VecRho[0]);
    tripletList.push_back(T(Nx-1,0,tempo));
    //lignes intermediaires 
    for (int i=1; i<Nx-1; i++){
        tempo = -(1.+VecRho[i-1]/VecRho[i]);
        tripletList.push_back(T(i-1,i,tempo));
        tempo = 2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i];
        tripletList.push_back(T(i,i,tempo));
        tempo = -(1.+VecRho[i+1]/VecRho[i]);
        tripletList.push_back(T(i+1,i,tempo));
    }
    tempo = -(1.+VecRho[0]/VecRho[Nx-1]);
    tripletList.push_back(T(0,Nx-1,tempo));
    tempo = -(1.+VecRho[Nx-2]/VecRho[Nx-1]);
    tripletList.push_back(T(Nx-2,Nx-1,tempo));
    tempo = 2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1];
    tripletList.push_back(T(Nx-1,Nx-1,tempo));
    Mat.setFromSortedTriplets(tripletList.begin(),tripletList.end());
    std::cout << "j'affiche la matrice dx z dx rho" << std::endl;
    std::cout << std::setprecision(5) << MatrixXd(Mat) << std::endl;
    return Mat;
}

Eigen::SparseMatrix<double> TermeDrhoDz(DataFile *df, double dt, Eigen::VectorXd(VecRho))
{
    //matrix D-E
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(Nx,Nx);
    typedef Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(3*Nx);
    double tempo;
    double lambda=dt/pow(dx,2);
    double cst = 1.;
    //ligne 1 
    tempo = 2.-(VecRho[1]+VecRho[Nx-1])/VecRho[0];
    tripletList.push_back(T(0,0,tempo));
    tempo = -(1.-VecRho[1]/VecRho[0]);
    tripletList.push_back(T(1,0,tempo));
    tempo = -(1.-VecRho[Nx-1]/VecRho[0]);
    tripletList.push_back(T(Nx-1,0,tempo));
    //lignes intermediaires 
    for (int i=1; i<Nx-1; i++){
        tempo = -(1.-VecRho[i-1]/VecRho[i]);
        tripletList.push_back(T(i-1,i,tempo));
        tempo = 2.-(VecRho[i-1]+VecRho[i+1])/VecRho[i];
        tripletList.push_back(T(i,i,tempo));
        tempo = -(1.-VecRho[i+1]/VecRho[i]);
        tripletList.push_back(T(i+1,i,tempo));
    }
    tempo = -(1.-VecRho[0]/VecRho[Nx-1]);
    tripletList.push_back(T(0,Nx-1,tempo));
    tempo = -(1.-VecRho[Nx-2]/VecRho[Nx-1]);
    tripletList.push_back(T(Nx-2,Nx-1,tempo));
    tempo = 2.-(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1];
    tripletList.push_back(T(Nx-1,Nx-1,tempo));
    Mat.setFromSortedTriplets(tripletList.begin(),tripletList.end());
    std::cout << "j'affiche la matrice dx rho dx z" << std::endl;
    std::cout << std::setprecision(5) << MatrixXd(Mat) << std::endl;
    return Mat;    
}

Eigen::SparseMatrix<double> BuildTest(DataFile *df, double dt, Eigen::VectorXd VecRho)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    double lambda = dt/pow(dx,2);
    Eigen::SparseMatrix<double> Mat(Nx,Nx);
    Eigen::SparseMatrix<double> DpE(Nx,Nx);
    Eigen::SparseMatrix<double> DmE(Nx,Nx);
    DpE = TermeDzDrho(df,dt,VecRho);
    DmE = TermeDrhoDz(df,dt,VecRho);
    Mat = 2*kappa*nu*DpE + 2*(1-kappa)*nu*DmE;
    return Mat;
}

Eigen::SparseMatrix<double> TeL(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //termes en laplacien -> termes diagonaux
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(6*Nx);
    double val0 = 1+2.*dt*nu/pow(dx,2);
    double valx = -nu*dt/pow(dx,2);
    //Carre 1 
    //ligne 1
    tL.push_back(T(0,0,val0));
    tL.push_back(T(1,0,valx));
    tL.push_back(T(Nx-1,0,valx));
    //lignes
    for (int i=1; i<Nx-1; i++){
        tL.push_back(T(i-1,i,valx));
        tL.push_back(T(i,i,val0));
        tL.push_back(T(i+1,i,valx));
    }
    //ligne N
    tL.push_back(T(0,Nx-1,valx));
    tL.push_back(T(Nx-2,Nx-1,valx));
    tL.push_back(T(Nx-1,Nx-1,val0));
    //Carre 2
    val0 = 1+2.*dt*nu*2.*kappa/pow(dx,2);
    valx = -nu*2*kappa*dt/pow(dx,2);
    //ligne 1
    tL.push_back(T(0+Nx,0+Nx,val0));
    tL.push_back(T(1+Nx,0+Nx,valx));
    tL.push_back(T(Nx-1+Nx,0+Nx,valx));
    //lignes
    for (int i=1; i<Nx-1; i++){
        tL.push_back(T(i-1+Nx,i+Nx,valx));
        tL.push_back(T(i+Nx,i+Nx,val0));
        tL.push_back(T(i+1+Nx,i+Nx,valx));
    }
    //ligne N
    tL.push_back(T(0+Nx,Nx-1+Nx,valx));
    tL.push_back(T(Nx-2+Nx,Nx-1+Nx,valx));
    tL.push_back(T(Nx-1+Nx,Nx-1+Nx,val0));

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> TeR(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //termes en r -> diagonale sur le premier bloc
    int Nx(df->getNx());
    double r(df->getR());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(Nx);
    double cst = dt*r;
    for (int i=0; i<Nx; i++){
        tL.push_back(T(i,i,cst));
    }
    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> TeD(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //termes sur les blocs extra-diagonaux
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(6*Nx);
    double hd = 0.5*dt/pow(dx,2)*(epsilon-4*kappa*(1-kappa)*pow(nu,2)/epsilon);
    double bg = -0.5*dt/pow(dx,2)*epsilon;
    double tempo;
    //bloc bas gauche
    // dx rho dx u -> rho en Dirichlet et u en Neumann
    if (cas==8){
        tempo = bg*(1+VecRho[1]/VecRho[0]+4*rho0(t,xmin,df)/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -bg*(3+VecRho[1]/VecRho[0]);
        tL.push_back(T(1+Nx,0,tempo));
    }
    if (cas==8){
        tempo = bg*(1+VecRho[1]/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -bg*(1+VecRho[1]/VecRho[0]);
        tL.push_back(T(1+Nx,0,tempo));
    }
    else {
        tempo = bg*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -bg*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(1+Nx,0,tempo));
        tempo = -bg*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1+Nx,0,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -bg*(1.+VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1+Nx,i,tempo));
        tempo = bg*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i+Nx,i,tempo));
        tempo = -bg*(1.+VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1+Nx,i,tempo));
    }
    if (cas==8){
        tempo = -bg*(3+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2+Nx,Nx-1,tempo));
        tempo = bg*(1+VecRho[Nx-2]/VecRho[Nx-1]+4*rho0(t,xmax,df)/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    else {
        tempo = -bg*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0+Nx,Nx-1,tempo));
        tempo = -bg*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2+Nx,Nx-1,tempo));
        tempo = bg*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    //bloc haut droit
    if (cas==8){
        tempo = hd*(1+VecRho[1]/VecRho[0]+4*rho0(t,xmin,df)/VecRho[0]);
        tL.push_back(T(0,0+Nx,tempo));
        tempo = -hd*(3+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0+Nx,tempo));
    }
    else {
        tempo = hd*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0,0+Nx,tempo));
        tempo = -hd*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0+Nx,tempo));
        tempo = -hd*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1,0+Nx,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -hd*(1.+VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1,i+Nx,tempo));
        tempo = hd*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i+Nx,tempo));
        tempo = -hd*(1.+VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1,i+Nx,tempo));
    }
    if (cas==8){
        tempo = -hd*(3+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1+Nx,tempo));
        tempo = hd*(1+VecRho[Nx-2]/VecRho[Nx-1]+4*rho0(t,xmax,df)/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1+Nx,tempo));
    }
    else {
        tempo = -hd*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0,Nx-1+Nx,tempo));
        tempo = -hd*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1+Nx,tempo));
        tempo = hd*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1+Nx,tempo));
        Mat.setFromSortedTriplets(tL.begin(),tL.end());
    }
    return Mat; 
}

Eigen::SparseMatrix<double> TekE(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //terme de la matrice E sur le premier bloc, peut être à changer.
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = dt/pow(dx,2)*nu*(1.-2.*kappa);
    double tempo;
    //bloc
    tempo = cst*((VecRho[1]+VecRho[Nx-1])/VecRho[0]);
    tL.push_back(T(0,0,tempo));
    tempo = -cst*(VecRho[1]/VecRho[0]);
    tL.push_back(T(1,0,tempo));
    tempo = -cst*(VecRho[Nx-1]/VecRho[0]);
    tL.push_back(T(Nx-1,0,tempo));
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1,i,tempo));
        tempo = cst*((VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i,tempo));
        tempo = -cst*(VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1,i,tempo));
    }
    tempo = -cst*(VecRho[0]/VecRho[Nx-1]);
    tL.push_back(T(0,Nx-1,tempo));
    tempo = -cst*(VecRho[Nx-2]/VecRho[Nx-1]);
    tL.push_back(T(Nx-2,Nx-1,tempo));
    tempo = cst*((VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
    tL.push_back(T(Nx-1,Nx-1,tempo));

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

void BuildMatG (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    Eigen::SparseMatrix<double> MTeL(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTeR(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTeD(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTekE(2*Nx,2*Nx);
    MTeL = TeL(df, dt, VecRho, t);
    MTeR = TeR(df, dt, VecRho, t);
    MTeD = TeD(df, dt, VecRho, t);
    MTekE = TekE(df, dt, VecRho, t);
    MatG = MTeL + MTeR + MTeD + MTekE;
    //std::cout << "j affiche matg" << std::endl;
    //std::cout << std::setprecision(6) << MatrixXd(MatG) << std::endl;
}

Eigen::SparseMatrix<double> TDpE(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //On fait les termes diagonaux pour avoir les bonnes conditions limites
    //On calcule les parties en D+E
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(6*Nx);
    double cst = dt/pow(dx,2)*nu*(1.-kappa);
    double tempo;
    //bloc 1
    if (cas==7){
        //premiere ligne avec cl dirichlet
        tempo = cst*(1+VecRho[1]/VecRho[0]+4*rho0(t,xmin,df)/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(3+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0,tempo));
    }
    else {
        //premier ligne periodique
        tempo = cst*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0,tempo));
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1,0,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1.+VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1,i,tempo));
        tempo = cst*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i,tempo));
        tempo = -cst*(1.+VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1,i,tempo));
    }
    if (cas==7){
        // derniere ligne avec cl dirichlet
        tempo = -cst*(3+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1,tempo));
        tempo = cst*(1+VecRho[Nx-2]/VecRho[Nx-1]+4*rho0(t,xmax,df)/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }
    else {
        // derniere ligne periodique
        tempo = -cst*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0,Nx-1,tempo));
        tempo = -cst*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1,tempo));
        tempo = cst*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }
    //bloc 2
    if (cas==7){
        //premiere ligne avec cl dirichlet
        tempo = cst*(1+VecRho[1]/VecRho[0]+4*rho0(t,xmin,df)/VecRho[0]);
        tL.push_back(T(0+Nx,0+Nx,tempo));
        tempo = -cst*(3+VecRho[1]/VecRho[0]);
        tL.push_back(T(1+Nx,0+Nx,tempo));
    }
    else {
        //premier ligne periodique
        tempo = cst*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0+Nx,0+Nx,tempo));
        tempo = -cst*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(1+Nx,0+Nx,tempo));
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1+Nx,0+Nx,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1.+VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1+Nx,i+Nx,tempo));
        tempo = cst*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i+Nx,i+Nx,tempo));
        tempo = -cst*(1.+VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1+Nx,i+Nx,tempo));
    }
    if (cas==7){
        // derniere ligne avec cl dirichlet
        tempo = -cst*(3+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2+Nx,Nx-1+Nx,tempo));
        tempo = cst*(1+VecRho[Nx-2]/VecRho[Nx-1]+4*rho0(t,xmax,df)/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1+Nx,tempo));
    }
    else {
        // derniere ligne avec cl dirichlet
        tempo = -cst*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0+Nx,Nx-1+Nx,tempo));
        tempo = -cst*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2+Nx,Nx-1+Nx,tempo));
        tempo = cst*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1+Nx,tempo));
    }

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> TDmE(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //On fait les termes diagonaux pour avoir les bonnes conditions limites
    //On calcule les parties en D-E
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(6*Nx);
    double cst = dt/pow(dx,2)*nu*(1.-2.*kappa);
    double tempo;

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> Thg(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //on calcule le bloc en haut à gauche en 2nu dx (rho dx u)
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = dt/pow(dx,2)*nu;
    double tempo;
    //bloc 1
    if (cas==8){
        //premiere ligne avec cl dirichlet
        tempo = cst*(1+VecRho[1]/VecRho[0]+4*rho0(t,xmin,df)/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(3+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0,tempo));
    }
    else {
        //premier ligne periodique
        tempo = cst*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(1,0,tempo));
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1,0,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1.+VecRho[i-1]/VecRho[i]);
        tL.push_back(T(i-1,i,tempo));
        tempo = cst*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i,tempo));
        tempo = -cst*(1.+VecRho[i+1]/VecRho[i]);
        tL.push_back(T(i+1,i,tempo));
    }
    if (cas==8){
        // derniere ligne avec cl dirichlet
        tempo = -cst*(3+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1,tempo));
        tempo = cst*(1+VecRho[Nx-2]/VecRho[Nx-1]+4*rho0(t,xmax,df)/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }
    else {
        // derniere ligne periodique
        tempo = -cst*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0,Nx-1,tempo));
        tempo = -cst*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-2,Nx-1,tempo));
        tempo = cst*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> Tbd(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //On calcule le terme en bas a droite en dxx rho v
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(6*Nx);
    double val0 = 2.*dt*nu/pow(dx,2);
    double valx = -nu*dt/pow(dx,2);
    //Carre 2
    val0 = 2.*dt*nu*2.*kappa/pow(dx,2);
    valx = -nu*2*kappa*dt/pow(dx,2);
    //ligne 1
    tL.push_back(T(0+Nx,0+Nx,val0));
    tL.push_back(T(1+Nx,0+Nx,valx));
    tL.push_back(T(Nx-1+Nx,0+Nx,valx));
    //lignes
    for (int i=1; i<Nx-1; i++){
        tL.push_back(T(i-1+Nx,i+Nx,valx));
        tL.push_back(T(i+Nx,i+Nx,val0));
        tL.push_back(T(i+1+Nx,i+Nx,valx));
    }
    //ligne N
    tL.push_back(T(0+Nx,Nx-1+Nx,valx));
    tL.push_back(T(Nx-2+Nx,Nx-1+Nx,valx));
    tL.push_back(T(Nx-1+Nx,Nx-1+Nx,val0));

    Mat.setFromSortedTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> TId(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //on rajoute la matrice identite
    int cas(df->getCas());
    int Nx(df->getNx());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(2*Nx);
    for (int i=0;i<2*Nx; i++){
        tL.push_back(T(i,i,1.));
    }
    //Mat.setFromSortedTriplets(tL.begin(),tL.end());
    Mat.setFromTriplets(tL.begin(),tL.end());
    return Mat;
}

Eigen::SparseMatrix<double> Thd2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //on calcule le bloc en haut à droite en (epsilon-4k(1-k)nunu/epsilon dx (rho dx v)
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = 0.5*(epsilon-4*kappa*(1-kappa)*pow(nu,2)/epsilon)*dt/pow(dx,2);
    double tempo;
    //premiere ligne
    if (cas==7||cas==8||cas==9){
        tempo = cst*(1.+(VecRho[1]+4.*rho0(t, xmin, df))/VecRho[0]);
        tL.push_back(T(0,0+Nx,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[1]);
        tL.push_back(T(0,1+Nx,tempo));
    }
    else {
        tempo = cst*(2 + (VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0,0+Nx,tempo));
        tempo = -cst*(1+VecRho[0]/VecRho[1]);
        tL.push_back(T(0,1+Nx,tempo));
        tempo = -cst*(1+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0,Nx-1+Nx,tempo));
    }
    //lignes internes
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1+VecRho[i]/VecRho[i-1]);
        tL.push_back(T(i,i-1+Nx,tempo));
        tempo = cst*(2+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i+Nx,tempo));
        tempo = -cst*(1+VecRho[i]/VecRho[i+1]);
        tL.push_back(T(i,i+1+Nx,tempo));
    }
    //derniere ligne
    if (cas==7||cas==8||cas==9){
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1,Nx-2+Nx,tempo));
        tempo = cst*(1+(VecRho[Nx-2]+4*rho0(t,xmax,df))/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1+Nx,tempo));
    }
    else {
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1,0+Nx,tempo));
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1,Nx-2+Nx,tempo));
        tempo = cst*(2+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1+Nx,tempo));
    }
    Mat.setFromTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> Tbg2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //on calcule le bloc en bas a gauche en -epsilon dx (rho dx u)
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = -0.5*epsilon*dt/pow(dx,2);
    double tempo;
    //premiere ligne
    #if 1
    if (cas==7||cas==8||cas==9){
        //tempo = cst*(1+VecRho[1]/VecRho[0]);
        tempo = cst*(VecRho[1]/VecRho[0]+1.);
        tL.push_back(T(0+Nx,0,tempo));
        //tempo = -cst*(1+VecRho[0]/VecRho[1]);
        tempo = -cst*(1.+VecRho[0]/VecRho[1]);
        tL.push_back(T(0+Nx,1,tempo));
    }
    else {
        tempo = cst*(2 + (VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -cst*(1+VecRho[0]/VecRho[1]);
        tL.push_back(T(0+Nx,1,tempo));
        tempo = -cst*(1+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0+Nx,Nx-1,tempo));
    }
    //lignes internes
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1+VecRho[i]/VecRho[i-1]);
        tL.push_back(T(i+Nx,i-1,tempo));
        tempo = cst*(2+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i+Nx,i,tempo));
        tempo = -cst*(1+VecRho[i]/VecRho[i+1]);
        tL.push_back(T(i+Nx,i+1,tempo));
    }
    //derniere ligne
    if (cas==7||cas==8||cas==9){
        //tempo = -cst*(1+VecRho[Nx-1]/VecRho[Nx-2]);
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1+Nx,Nx-2,tempo));
        //tempo = cst*(1+VecRho[Nx-2]/VecRho[Nx-1]);
        tempo = cst*(VecRho[Nx-2]/VecRho[Nx-1]+1.);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    else {
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1+Nx,0,tempo));
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1+Nx,Nx-2,tempo));
        tempo = cst*(2+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    #else 
    //bloc 1
    if (cas==7||cas==8||cas==9){
        tempo = cst*(1+VecRho[1]/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -cst*(1+VecRho[0]/VecRho[1]);
        tL.push_back(T(0+Nx,1,tempo));
    }
    else {
        //premier ligne periodique
        tempo = cst*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0+Nx,0,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[1]);
        tL.push_back(T(0+Nx,1,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0+Nx,Nx-1,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1.+VecRho[i]/VecRho[i-1]);
        tL.push_back(T(i+Nx,i-1,tempo));
        tempo = cst*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i+Nx,i,tempo));
        tempo = -cst*(1.+VecRho[i]/VecRho[i+1]);
        tL.push_back(T(i+Nx,i+1,tempo));
    }
    if (cas==7||cas==8||cas==9){
        tempo = -cst*(1+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1+Nx,Nx-2,tempo));
        tempo = cst*(1+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    else {
        // derniere ligne periodique
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1+Nx,0,tempo));
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1+Nx,Nx-2,tempo));
        tempo = cst*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1+Nx,Nx-1,tempo));
    }
    #endif
    Mat.setFromTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> Thg2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //on calcule le bloc en haut à gauche en 2nu dx (rho dx u)
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double nu(df->getNu());
    double epsilon(df->getEpsilon());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double cst = dt/pow(dx,2)*nu;
    double tempo;
    //bloc 1
    if (cas==7||cas==8||cas==9){
        tempo = cst*(1.+VecRho[1]/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[1]);
        tL.push_back(T(0,1,tempo));
    }
    else {
        //premier ligne periodique
        tempo = cst*(2.+(VecRho[1]+VecRho[Nx-1])/VecRho[0]);
        tL.push_back(T(0,0,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[1]);
        tL.push_back(T(0,1,tempo));
        tempo = -cst*(1.+VecRho[0]/VecRho[Nx-1]);
        tL.push_back(T(0,Nx-1,tempo));
    }
    for (int i=1; i<Nx-1; i++){
        tempo = -cst*(1.+VecRho[i]/VecRho[i-1]);
        tL.push_back(T(i,i-1,tempo));
        tempo = cst*(2.+(VecRho[i-1]+VecRho[i+1])/VecRho[i]);
        tL.push_back(T(i,i,tempo));
        tempo = -cst*(1.+VecRho[i]/VecRho[i+1]);
        tL.push_back(T(i,i+1,tempo));
    }
    if (cas==7||cas==8||cas==9){
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1,Nx-2,tempo));
        tempo = cst*(1.+VecRho[Nx-2]/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }
    else {
        // derniere ligne periodique
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[0]);
        tL.push_back(T(Nx-1,0,tempo));
        tempo = -cst*(1.+VecRho[Nx-1]/VecRho[Nx-2]);
        tL.push_back(T(Nx-1,Nx-2,tempo));
        tempo = cst*(2.+(VecRho[0]+VecRho[Nx-2])/VecRho[Nx-1]);
        tL.push_back(T(Nx-1,Nx-1,tempo));
    }

    //Mat.setFromSortedTriplets(tL.begin(),tL.end());
    Mat.setFromTriplets(tL.begin(),tL.end());
    return Mat; 
}

Eigen::SparseMatrix<double> Tbd2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    //On calcule le terme en bas a droite en 2kappa nu* (dxx rho v)
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    Eigen::SparseMatrix<double> Mat(2*Nx,2*Nx);
    typedef Triplet<double> T;
    std::vector<T> tL;
    tL.reserve(3*Nx);
    double val0 = 2*kappa*2.*dt*nu/pow(dx,2);
    double valb = 2*kappa*3.*dt*nu/pow(dx,2);
    double valx = -2*kappa*nu*dt/pow(dx,2);
    //Carre 2
    val0 = 2.*dt*nu*2.*kappa/pow(dx,2);
    valx = -nu*2*kappa*dt/pow(dx,2);
    valb = 3.*dt*nu*2.*kappa/pow(dx,2);
    //ligne 1
    if (cas==7||cas==8||cas==9){
        tL.push_back(T(0+Nx,0+Nx,valb));
        tL.push_back(T(0+Nx,1+Nx,valx));
    }
    else {
        tL.push_back(T(0+Nx,0+Nx,val0));
        tL.push_back(T(0+Nx,1+Nx,valx));
        tL.push_back(T(0+Nx,Nx-1+Nx,valx));
    }
    //lignes
    for (int i=1; i<Nx-1; i++){
        tL.push_back(T(i+Nx,i-1+Nx,valx));
        tL.push_back(T(i+Nx,i+Nx,val0));
        tL.push_back(T(i+Nx,i+1+Nx,valx));
    }
    //ligne N
    if (cas==7||cas==8||cas==9){
        tL.push_back(T(Nx-1+Nx,Nx-2+Nx,valx));
        tL.push_back(T(Nx-1+Nx,Nx-1+Nx,valb));
    }
    else {
        tL.push_back(T(Nx-1+Nx,Nx,valx));
        tL.push_back(T(Nx-1+Nx,Nx-2+Nx,valx));
        tL.push_back(T(Nx-1+Nx,Nx-1+Nx,val0));
    }
    Mat.setFromTriplets(tL.begin(),tL.end());
    return Mat; 
}

void BuildMatG2 (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    Eigen::SparseMatrix<double> MTId(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTeR(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTeD(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MThg(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTbd(2*Nx,2*Nx);
    MTId = TId(df, dt, VecRho, t);
    MTeR = TeR(df, dt, VecRho, t);
    MTeD = TeD(df, dt, VecRho, t);
    MThg = Thg(df, dt, VecRho, t);
    MTbd = Tbd(df, dt, VecRho, t);
    MatG = MTId + MTeR + MTeD + MThg + MTbd;
    //std::cout << "j affiche matg2" << std::endl;
    //std::cout << std::setprecision(6) << MatrixXd(MatG) << std::endl;
}

void BuildMatG3 (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    Eigen::SparseMatrix<double> MTId(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTeR(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MThd(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MThg(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTbg(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MTbd(2*Nx,2*Nx);
    MTId = TId(df, dt, VecRho, t);
    MTeR = TeR(df, dt, VecRho, t);
    MThg = Thg2(df, dt, VecRho, t);
    MThd = Thd2(df, dt, VecRho, t);
    MTbg = Tbg2(df, dt, VecRho, t);
    MTbd = Tbd2(df, dt, VecRho, t);
    MatG = MTId+ MTeR + MThg + MThd + MTbg + MTbd;
    //std::cout << "j affiche matg3" << std::endl;
    //std::cout << std::setprecision(6) << MatrixXd(MatG) << std::endl;

}

void Print_Psi(Eigen::VectorXd VecPsi, double t, DataFile *df, int pdt)
{
    int cas(df->getCas());
    int Nx(df->getNx());
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx=(xmax-xmin)/Nx;
    double x, xp;
    double c=1.;
    double tmax(df->getTmax());
    //double t=tmax;
    double err = 0.;
    double psit;
    Eigen::VectorXd ErrPsi(Nx);
    if (cas==8){
        for (int i=0; i<Nx; i++){
            x = xmin+0.5*dx+i*dx;
            psit=Psi0(t,x,df);
            ErrPsi[i] = psit-VecPsi[i];
        }
        for (int i=0; i<Nx; i++){
            err += pow(ErrPsi[i],2)*dx;
        }
    }

    std::string resultatChemin(df->getResultatChemin());
    std::string filename;
    filename = resultatChemin+"Psi"+std::to_string(pdt)+".dat";
    std::ofstream sol_file(filename);
    sol_file << "l'erreur en psi = " << sqrt(err) << std::endl;
    for (int i=0; i<Nx; i++){
        x = xmin+0.5*dx+i*dx;
        xp = x-c*t+(xmax-xmin)*(1.+floor((c*t-x+xmin)/(xmax-xmin)));
        sol_file << x << " " << VecPsi[i] << " " << Psi0(t,x,df) << std::endl;
    }
}

void InitFromFiles(Eigen::VectorXd &vecU, DataFile *df)
{
    //On veut faire une initialisation à partir d'un fichier de données
    //a finir
}
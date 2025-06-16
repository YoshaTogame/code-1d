#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <list>
#include <omp.h>
#include <iomanip>

#include "Dense"
#include "Sparse"
#include "IterativeLinearSolvers"
#include "DataFile.h"
#include "Fonction.h"
#include "SparseLU"



using namespace Eigen;

int main(int argc, char **argv){
    if (argc < 2)
    {
        std::cout << "Please, enter the name of your data file." << std::endl;
        exit(0);
    }
    const std::string data_file_name = argv[1];
    //int VraiNx = atoi(argv[2]);
    // Fichier ou on va ecrire la solution
    std::string resultat, residus;
    DataFile *df = new DataFile(data_file_name);
    df->litLeFichier();
    int debug = df->getDebug();
    //df->changeNx(VraiNx);
    //std::cout << VraiNx << " " << df->getNx() << std::endl;
    int nbImg = df->getNbImg();
    int nbPts = df->getNbPts();
    int cas = df->getCas();
    double tmax = df->getTmax();
    double r = df->getR();
    double cfl = df->getCFL();
    int Nx = df->getNx();
    double xmax(df->getXMax());
    double xmin(df->getXMin());
    double dx = (xmax-xmin)/Nx;
    double nu(df->getNu());
    double kappa(df->getKappa());
    double gamma(df->getGamma());
    int pdt=0;
    int incr=0; 
    int maxincr = 5;
    double dt;
    double dtempo;
    double tempsSortie=0.;
    bool affiche = false;
    std::cout << "tmax =" << tmax << " cas = "<< cas << " Nx = " << Nx << std::endl;
    std::vector<double> vecRho(Nx);
    Eigen::VectorXd vecU(3*Nx);
    Eigen::VectorXd rho(Nx);
    Eigen::VectorXd Flux(3*Nx);
    Eigen::VectorXd VecTempo(Nx);
    Eigen::VectorXd UV(2*Nx);
    Eigen::VectorXd VecTempoUV(2*Nx);
    Eigen::VectorXd VecC(Nx);
    Eigen::VectorXd VecV(Nx);
    Eigen::VectorXd PSM(Nx);
    Eigen::VectorXd source2(2*Nx);
    Eigen::VectorXd Source(2*Nx);
    Eigen::VectorXd SourceDroit(2*Nx);
    Eigen::VectorXd SourceRho(Nx);
    Eigen::VectorXd TamponSourceRho(Nx);
    Eigen::SparseMatrix<double> D(Nx,Nx);
    Eigen::SparseMatrix<double> LPsi(Nx,Nx);
    Eigen::BiCGSTAB<SparseMatrix<double>> solverD;
    Eigen::SparseMatrix<double> DC(Nx,Nx);
    BuildLaplacian(DC,df,1.);
    Eigen::SparseMatrix<double> M(2*Nx,2*Nx);
    Eigen::SparseMatrix<double> MatG(2*Nx,2*Nx);
    Eigen::BiCGSTAB<SparseMatrix<double>, Eigen::IncompleteLUT<double>> solverM;
    Eigen::SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solverM2;
    Initialize(vecU, df);
    std::cout << "on a fait initialize" << std::endl;
    LPsi = LaplacianPsi(df);
    std::cout << "on a fait le laplacien" << std::endl;
    saveSol(df, vecU, pdt);
    tempsSortie = tmax/nbImg;
    std::list<double> energieList;
    std::vector<double> energieVec(nbImg+1);
    std::vector<double> tempsVec(nbImg+1);
    bool valbool = true;
    energieList.insert(energieList.end(),calculEnergie(vecU, df));
    double time = 0.;
    double time2 = 0.;
    energieVec[pdt] = calculEnergie(vecU, df);
    tempsVec[pdt] = pdt;
    while (time<tmax && valbool || incr < maxincr)
    {
        if (cas!=9){
            maxincr = -1;
        }
        if (debug==1){
            std::cout << "je suis dans le while" << std::endl;
        }
        //valbool = false;
        /*
        dt = 2*vecU[Nx]/vecU[0]+sqrt(vecU[0]);
        for (int i=1; i<Nx; i++){
            dtempo = 2*(vecU[Nx+i]/vecU[0+i]+sqrt(vecU[0+i]));
            if (dt<dtempo){
                dt=dtempo;
            }
        }
        dt = cfl*dx/dt;
        */

        // je refais le pas de temps au cas ou on avait un soucis de cfl
        dt = vecU[0+Nx]/vecU[0] + sqrt(gamma*pow(vecU[0],gamma-1)+2*r*kappa*nu);
        for (int i=1; i<Nx; i++){
            dtempo = vecU[i+Nx]/vecU[i] + sqrt(gamma*pow(vecU[i],gamma-1)+2*r*kappa*nu);
            if (dt<dtempo){
                dt=dtempo;
            }
        }
        dt = cfl*dx/dt;

        if (time+dt > tempsSortie){
            dt = tempsSortie-time;
            time2=time;
            time = tempsSortie;
            pdt = pdt+1;
            if (cas==9){
                tempsSortie += tmax/nbImg;
                
            }
            else {
                tempsSortie = std::min(tempsSortie+tmax/nbImg,tmax);
            }
            affiche = true;
            std::cout << "time = " << time << " ,dt = " << dt << " ,pdt =" << pdt << std::endl;
        }
        else {
            time2 = time;
            time = time+dt;
        }
        //affiche=true;
        if (debug==1){
            std::cout << "le pas de temps est dt=" << dt << std::endl;
        }
        //phase hyperbolique
        FluxRusanov2(Flux, df, vecU, time2);
        //std::cout << "Flux = " << Flux << std::endl;
        updateRusanov2(vecU,Flux,df, dt, time2);
        //std::cout << "sans source, vecU =" << vecU << std::endl;
        if (debug==1){
            std::cout << "on a fait la partie hyperbolique" << std::endl;
        }
        //phase diffusive

        //je rajoute le terme source dans la partie hyperbolique
        #if 0
        SourceRho = dt*BuildSourceRho(df,time2);
        //std::cout << "source rho =" << SourceRho << std::endl;
        source2 = dt*BuildSource2(rho,VecV,df,dt,time2);
        //std::cout << "source2 =" << source2 << std::endl;
        for (int i=0; i<Nx; i++){
            vecU[i] += SourceRho[i];
            vecU[i+Nx] += source2[i];
            vecU[i+2*Nx] += source2[i+Nx];
        }
        //std::cout << "vec U avec le terme source = " << vecU << std::endl;
        #endif 

        #if 1 //on fait la partie diffusive (sinon que la partie hyperbolique)
        if (cas==1){
            //conditions de Dirichlet
            double lambda = dt/(dx*dx);
            VecTempo[0] = vecU[0] + 2.*2.*kappa*lambda*nu  ;
            for (int i=1; i<Nx-1; i++){
            VecTempo[i] = vecU[i];
            }
            VecTempo[Nx-1] = vecU[Nx-1] + 2.*kappa*lambda*nu;
        }
        else {
            for (int i=0; i<Nx; i++){
            VecTempo[i] = vecU[i];
            }
        }
        BuildLaplacian(D,df,dt);
        D.makeCompressed();
        if (debug==1){
            std::cout << "le laplacien est construit" << std::endl;
        }
        solverD.compute(D);
        SourceRho = dt*BuildSourceRho(df,time);
        TamponSourceRho = SourceRho + VecTempo;
        //rho = solverD.solve(VecTempo);
        rho = solverD.solve(TamponSourceRho);
        for (int i=0; i<Nx; i++){
            vecRho[i] = rho[i];
        }
        if (cas==8||cas==9||cas==10){
            //on ajoute le potentiel
            //BuildV(VecV, time, rho, LPsi, incr, df);
        }
        if (debug==1){
            std::cout << "on a fait le terme source 2 en rho grad v" << std::endl;
        }
        //on a fini avec le potentiel electrique
        if (cas==1){
            Build_Matrix(M,df,dt,rho);
        }
        else {
            //BuildMatG(MatG,df,dt,rho,time);
            //BuildMatG2(MatG,df,dt,rho,time);
            BuildMatG3(MatG,df,dt,rho,time);
            M = MatG;
        }
        //BuildMatG(MatG,df,dt,rho);
        //Build_Matrix(M,df,dt,rho);
        //M = MatG;
        //std::cout << "jaffiche M" << std::endl;
        //std::cout << std::setprecision(6) << MatrixXd(M) << std::endl;
        M.makeCompressed();
        if (debug==1){
            std::cout << "on a construit la matrice M" << std::endl;
        }
        for (int i=0; i<Nx; i++){
            double x;
            x = xmin+i*dx+0.5*dx;
            VecTempoUV[i] = vecU[i+Nx];
            //VecTempoUV[i] = rho0(time,x,df)*w0(time,x,df);
            VecTempoUV[i+Nx] = vecU[i+2*Nx];
            //VecTempoUV[i+Nx] = rho0(time,x,df)*v0(time,x,df);
        }
        //on choisi le solver
        #if 0
        solverM.setTolerance(1.e-12);
        solverM.compute(M);
        UV = solverM.solve(VecTempoUV);
        #else 
        solverM2.analyzePattern(M);
        solverM2.factorize(M);
        source2 = dt*BuildSource2(rho,VecV,df,dt,time);
        SourceDroit = VecTempoUV + source2;
        //SourceDroit = VecTempoUV;
        if (debug==1){
            std::cout << "on a construit le terme source" << std::endl;
        }
        //UV = solverM2.solve(VecTempoUV);
        UV = solverM2.solve(SourceDroit);
        if (debug==1){
            std::cout << "on a inverse la matrice" << std::endl;
        }
        #endif
        for (int i=0; i<Nx; i++){
            double x;
            x = xmin+i*dx+0.5*dx;
            vecU[i] = rho[i];
            //vecU[i] = vecRho[i];
            vecU[i+Nx] = UV[i];
            //vecU[i+Nx] = vecRho[i]*w0(time,x,df);
            vecU[i+2*Nx] = UV[i+Nx];
            //vecU[i+2*Nx] = vecRho[i]*v0(time,x,df);
        }
        #endif
        energieList.insert(energieList.end(),calculEnergie(vecU, df));
        if(affiche){
            affiche = false;
            //saveSol(df,vecU,pdt);
            saveSol2(df,vecU,VecV,pdt);
            energieVec[pdt-1] = calculEnergie(vecU, df);
            //Print_Psi(VecV, time, df, pdt);
            tempsVec[pdt-1] = time;
            if (debug==1){
                std::cout << "on a ecrit la solution a t=" << time << std::endl;
            }
        }
        if (cas==9){
            //on regarde pour incrementer V à droite
            double maxj,minj,meanj;
            maxj = abs(vecU[Nx]);
            minj = vecU[Nx];
            meanj = vecU[Nx];
            for (int i=1; i<Nx; i++){
                if (vecU[Nx+i]>maxj){
                    maxj = vecU[Nx+i];
                }
                if (vecU[Nx+1]<minj){
                    minj = vecU[Nx+1]; 
                }
                meanj += vecU[Nx+i]/Nx;
            }
            if (abs(maxj-minj)/meanj < 1e-2){
                incr+=1;
                std::cout << "on met a jour le potentiel, avec incr = " << incr << std::endl;
            }
        }
    }
    saveEnergie(df, energieVec,  tempsVec);
    saveExacte(df,vecU,tmax);
    erreurL2(vecU, df, tmax);
    Print_Psi(VecV, time,df, 0);
    return 0;
}

#include <iostream>
#include <cmath>
#include "DataFile.h"
#include "Dense"
#include "Sparse"

void Build_Matrix(Eigen::SparseMatrix<double> &M, DataFile *df, double dt, Eigen::VectorXd vecRho);

void Initialize(Eigen::VectorXd &vecU, DataFile *df );

double rho0(double t,double x,DataFile *df);

double v0(double t,double x,DataFile *df);

double w0(double t,double x,DataFile *df);

double intRho0(double t, double x, DataFile *df);

double Psi0(double t, double x, DataFile *df);

void saveSol(DataFile *df, Eigen::VectorXd &vecU, int pdt);

void saveEnergie(DataFile *df, std::vector<double> energieVec, std::vector<double> tempsVec);

void FluxRusanov(Eigen::VectorXd &Flux, DataFile *df, Eigen::VectorXd &vecU, double t);

void FluxRusanov2(Eigen::VectorXd &Flux, DataFile *df, Eigen::VectorXd &vecU, double t);

void FluxMUSCL(Eigen::VectorXd &Flux, DataFile *df, Eigen::VectorXd &vecU, double t, int pdt, bool ecrit);

void updateRusanov(Eigen::VectorXd &vecU, Eigen::VectorXd &Flux, DataFile *df, double dt, double t);

void updateRusanov2(Eigen::VectorXd &vecU, Eigen::VectorXd &Flux, DataFile *df, double dt, double t);

void BuildLaplacian(Eigen::SparseMatrix<double> &D, DataFile *df, double dt);

double calculEnergie(Eigen::VectorXd &vecU, DataFile *df);

void BuildSource(Eigen::VectorXd &source, Eigen::VectorXd &rho, DataFile *df, double dt, double temps);

Eigen::SparseMatrix<double> LaplacianPsi(DataFile *df);

void DefDoping(Eigen::VectorXd &vecC, DataFile *df, double t);

void PoissonSM(Eigen::VectorXd &PSM, Eigen::VectorXd &rho, Eigen::VectorXd &vecC, DataFile *df);

void BuildV(Eigen::VectorXd &vecV, double t, Eigen::VectorXd &rho, Eigen::SparseMatrix<double> DC, int incr, DataFile *df);

Eigen::VectorXd BuildSource2(Eigen::VectorXd &rho, Eigen::VectorXd &vecV, DataFile *df, double dt, double t);

Eigen::VectorXd BuildSourceRho(DataFile *df, double t);

void saveSol2(DataFile *df, Eigen::VectorXd &vecU, Eigen::VectorXd &vecV, int pdt);

void saveExacte(DataFile *df, Eigen::VectorXd &vecU, double t);

void erreurL2(Eigen::VectorXd VecU, DataFile *df, double t);

Eigen::SparseMatrix<double> TermeDzDrho(DataFile *df, double dt, Eigen::VectorXd(VecRho));

Eigen::SparseMatrix<double> TermeDrhoDz(DataFile *df, double dt, Eigen::VectorXd(VecRho));

Eigen::SparseMatrix<double> BuildTest(DataFile *df, double dt, Eigen::VectorXd VecRho);

Eigen::SparseMatrix<double> TeL(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> TeR(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> TeD(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> TekE(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

void BuildMatG (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Thg(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Tbd(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> TId(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Thd(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Tbg(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Thg2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

Eigen::SparseMatrix<double> Tbd2(DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

void BuildMatG2 (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

void BuildMatG3 (Eigen::SparseMatrix<double> &MatG, DataFile *df, double dt, Eigen::VectorXd VecRho, double t);

void Print_Psi(Eigen::VectorXd VecPsi, double t, DataFile *df, int pdt);

void InitFromFiles(Eigen::VectorXd &vecU, DataFile *df);

double minmod(double a, double b, int i, Eigen::VectorXd & pente);
# Compilateur utilisé
CC=g++
#CC=mpicxx

CXX_FLAGS = -std=c++17 -I Eigen/Eigen  

# Options en mode optimisé - La variable DEBUG est définie comme fausse
OPTIM_FLAG = -O3 -DNDEBUG -std=c++17
# Options en mode debug - La variable est DEBUG est définie comme vraie
DEBUG_FLAG = -O0 -DDEBUG -pedantic -std=c++17



# On choisit comment on compile en ajoutant le debut ou l optim
CXX_FLAGS += $(OPTIM_FLAG)

# Le nom de l'exécutable
PROG = run.exe

# Les fichiers source à compiler
#SRC = src/main.cc  src/DataFile.cpp src/Algebra.cpp src/Solver.cpp src/Fonction.cpp src/Schwarz.cpp
SRC = src/main.cc src/DataFile.cpp src/Fonction.cpp
# La commande complète : compile seulement si un fichier a été modifié
$(PROG) : $(SRC)
	$(CC) $(SRC) $(CXX_FLAGS) -o $(PROG)
# Évite de devoir connaitre le nom de l'exécutable
all : $(PROG)

# Supprime l'exécutable, les fichiers binaires (.o) et les fichiers
# temporaires de sauvegarde (~)
clean :
	rm -f *.o *~ $(PROG)

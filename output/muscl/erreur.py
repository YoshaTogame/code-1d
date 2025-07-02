import numpy as np
import matplotlib.pyplot as plt

def lit_fichier(fileName,n):
    fichier = open(fileName)
    x = []
    rho = []
    u = []
    v = []
    for i in range (n):
        L = fichier.readline()
        Ls = L.split(" ")
        x.append(float(Ls[0]))
        rho.append(float(Ls[1]))
        u.append(float(Ls[2]))
        v.append(float(Ls[3]))
    fichier.close
    return [x,rho,u,v]

def rho(x):
    #print(x)
    rho = 1.5 -0.5/(np.power(np.cosh(np.sqrt(0.5)*x),2))
    return rho

def u(x):
    u = 2.-1.5/rho(x)
    return u

def v(x):
    dxrho = np.sinh(np.sqrt(0.5)*x)/np.power(np.cosh(np.sqrt(0.5)*x),3)
    dxrho = 2.*np.power(0.5,1.5)*dxrho
    v = 0.5*dxrho/rho(x)
    return v

def calcul_erreur(n):
    filename = str(n)+"/sol101.dat"
    vec = lit_fichier(filename,n)
    x = np.array(vec[0][:])
    rhon = vec[1][:]
    un = vec[2][:]
    vn = vec[3][:]
    rhoe = rho(x)
    ue = u(x)
    ve = v(x)
    #sh = np.sqrt(1/n)
    sh = 40./n
##    if (n==100):
##        print(rhon)
##        print(rhoe)
##        print(rhon-rhoe)
##        print(np.max(rhon-rhoe))
    errrho = sh*np.sum(np.power(np.array(rhon)-np.array(rhoe),2))
    erru = sh*np.sum(np.power(np.array(un)-np.array(ue),2))
    errv = sh*np.sum(np.power(np.array(vn)-np.array(ve),2))
    return [np.sqrt(errrho),np.sqrt(erru),np.sqrt(errv)]

e10 = calcul_erreur(100)
e20 = calcul_erreur(200)
e50 = calcul_erreur(400)
e200 = calcul_erreur(800)
e400 = calcul_erreur(1600)
xe = [100,200,400,800,1600]
y1e = [e10[0],e20[0],e50[0],e200[0],e400[0]]
y2e = [e10[1],e20[1],e50[1],e200[1],e400[1]]
y3e = [e10[0],e10[0]/2,e10[0]/4,e10[0]/8,e10[0]/16]
y4e = [e10[0],e10[0]/4,e10[0]/16,e10[0]/(4**3),e10[0]/(4**4)]

plt.xlabel("log(N)")
plt.ylabel("log(err)")
plt.plot(np.log(xe),np.log(y1e),marker="o",linestyle="-",color="b",label="err rho")
plt.plot(np.log(xe),np.log(y2e),marker="o",linestyle="-",color="r",label="err u")
plt.plot(np.log(xe),np.log(y3e),marker="o",linestyle="-",color="g",label="order 1")
plt.plot(np.log(xe),np.log(y4e),marker="o",linestyle="-",color="pink",label="order 2")
plt.title("error for different mesh size")
plt.legend()
plt.show(block=False)

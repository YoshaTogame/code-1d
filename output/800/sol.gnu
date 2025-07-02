# Définir le terminal pour la sortie EPS
set terminal postscript eps enhanced color


# Définir les titres et les labels
set title "numerical solution for different times"
set xlabel "x"
set ylabel "rho"

set xrange[-170:170]

# Tracer le premier graphique
plot 'sol0.dat' with lines title 't=0'
replot 'sol1.dat' w l title "t=7s"
replot 'sol3.dat' w l title "t=21s"
replot 'sol5.dat' w l title "t=35s"
set output 'sol.eps'
replot 'sol7.dat' w l title "t=49s"

# Réinitialiser le terminal pour revenir à l'affichage à l'écran
set terminal wxt


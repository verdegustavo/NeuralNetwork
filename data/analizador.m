J = load('costo.mat','-ascii');
i = 1:size(J)(1);
plot(i,J)
title("Gráfica de Costo por Iteraciones del GD", "fontsize", 15)
xlabel("# iteraciones")
ylabel("Costo J")
pause;

y = load('y.dat','-ascii');
p = load('prediccion.mat','-ascii');
fprintf('\nExactitud del Set de Entrenamiento: %f %%\n', mean(double(p== y)) * 100);

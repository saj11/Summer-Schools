[X,Y] = meshgrid(0:0.1:10,0:0.1:10);
%dot product equivalency
Z = X.^2  + Y.^2;
%figure; surf(X, Y, Z)
%gradient (manual)
%component 1
Xn = 2 * X;
%component 2
Yn = 2* Y;

%gradient lib
[FX ,FY] = gradient(Z, 0.1);
%gradient in X
figure; surf(Yn)
%figure; surf(Xn)

%gradient in Y
figure; surf(FY)
%figure; surf(Yn)



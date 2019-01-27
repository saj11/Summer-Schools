function curveFittingExample
    %N, number of observations
     Max = 1;
     N = 15;
     x = 0:Max/N:Max;
     
     %t_bold, set of target values
     [t, yReal] = getObservation(x);
     
     figure; scatter(x, t);
     hold on;
     %M, degree of polynomial to fit
     M = 300;
     w = getOptimumW(x, t, M);
     lambda = 0.0000;
     wReg = getOptimumW_Reg(x, t, M, lambda);
     for i = 1:length(x)
        y_approx(i) = yF(x(i), w);
        y_approxReg(i) = yF(x(i), wReg);
     end
     plot(x, y_approxReg);
     hold on;
     plot(x, yReal);
     [E, Erms] = getError(yReal, y_approx);
     Erms
end
%E(w)
function [E, Erms] = getError(y, t)
    E = 0.5 * sum((y - t).^2);
    Erms = sqrt((2 * E)/length(t));
end




%Least squares with first derivative equals zero
function W = getOptimumW_Reg(x, t, M, lambda)
    M = M + 1;
    t = t';
    N = length(x);
    X = zeros(length(x), M); 
    %Matrix X with x repeated in every column
    %to express the derivative in matrix therms
    for i = 1:M
        X(:, i) = x;
    end
    for i = 1:N
        for j = 1:M
            X(i, j) = X(i,j) ^ (j-1);
        end
    end
    
    term1 =   X' * X;
    lambdaI = lambda * eye(size(term1));
    fac1 = term1 + lambdaI;
    fac2 = X' * t;
    invFac1 = pinv(fac1);
    W = invFac1 * fac2;
    
end

%Least squares with first derivative equals zero
function W = getOptimumW(x, t, M)
    c = 1;
    %build matrices A and B
    %iteration for each row
    for m = M : 2*M
		%iterates each equation i
        B(c,1) = sum(t .* x .^ (c - 1));
        w = 1;
		%1.m = M
		%2.m = M + 1
        %iteration for each column
        for k = m - M:m
			%1.1 k = M-M=0
			%1.2 k = 1
			%1.3 k = 2 ....
			%2.1
            A(c, w) = sum(x .^ k);
            w = w + 1;
        end
        c = c + 1;
    end
    %solve the linear system, si hay singularidades puede reventar
    W = linsolve(A,B);
end

function [t, yReal] = getObservation(x)
    snr = 2;
    yReal = sin(2*pi*x);
    t = awgn(yReal, snr);
end

%polynomial approx
function y = yF(x_val, w)
    y = 0;
    for i = 1:length(w)
        y = y + w(i) * (x_val.^(i - 1));
    end
end
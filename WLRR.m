function [Z,D,E] = WLRR(X, Dinit, param)
%-------------------------------------------------------------------------------------
% Jul. 2017
%
% by He-Feng Yin, yinhefeng@126.com
% solve the optimization problem (5) in our paper
% min_{Z,E,D} ||Z||_* + \lambda ||E||_21 + \alpha ||W.*Z||_1 + \frac{\gamma}{2}||D||_F^2
% s.t. X = DZ+E.

% input
% X: training data, each column is a sample, with the size d*n
% Dinit: initialized dictionary, each column is an atom, with the size d*m
% param: a struct contains parameters
      % param.lambda: regularization term on the error matrix
      % param.alpha: regularization term on the weighted l_1
      % param.gamma: regularization term on dictionary
      % param.dist: the weight matrix


% output
% Z: the learned representation of X on D
% D: the learned dictionary
% E: the updated error matrix
%-------------------------------------------------------------------------------------

lambda = param.lambda;
alpha = param.alpha;
gamma = param.gamma;
W = param.dist;

%------------------------------------------------
% Paramters initialization
%------------------------------------------------
[d,n] = size(X);
[~,m] = size(Dinit);
tol = 1e-6;
maxIter = 1e4;
rho = 1.15;
mu = sqrt(max(d,m))\1;
max_mu = 1e8;
Z = zeros(m,n);
E = zeros(size(X));
J = zeros(m,n);
L = J;
Y1 = zeros(d,n);
Y2 = zeros(m,n);
Y3 = zeros(m,n);

% Start main loop
iter = 0;
D = Dinit;
while iter<maxIter
    iter = iter + 1;
    
    %update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    %------------------------------------------------
    % Update Z
    %------------------------------------------------
    Z_left = D'*D+2*eye(m);
    Z = Z_left \ ( D'*(X-E)+J+L+(D'*Y1 - Y2 - Y3)/mu);
    clear Z_left;
    
    %update L
    L_temp = Z+Y3/mu;
    B = mu\alpha*W;
    L = max(L_temp- B, 0)+min( L_temp+ B, 0);
    
    %------------------------------------------------
    % Update E
    %------------------------------------------------
    xmaz = X-D*Z;
    temp = xmaz+Y1/mu;
    %||E||_1
    %     E = max(temp- lambda/mu, 0)+min( temp+ lambda/mu, 0);
    
    %||E||_2,1
    for i=1:n
        temp2=temp(:,i);
        nw=norm(temp2);
        if nw>lambda/mu
            E(:,i)=(nw-lambda/mu)*temp2/nw;
        else
            E(:,i)=zeros(length(temp2),1);
        end
    end
    
    %update D
    D_trans= ( Z*Z'+gamma/mu*eye(m) ) \ ( Y1*Z'/mu - (E-X)*Z' )';
    D = D_trans';
    
    %------------------------------------------------
    % Convergence Validation
    %------------------------------------------------
    leq1 = xmaz-E;
    leq2 = Z-J;
    leq3 = Z-L;
    stopC1 = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(stopC1,max(max(abs(leq3))));
    if stopC<tol || iter>=maxIter
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
    if (iter==1 || mod(iter, 50 )==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    obj(iter) = norm(leq1,'fro')/norm(X,'fro');
end
end

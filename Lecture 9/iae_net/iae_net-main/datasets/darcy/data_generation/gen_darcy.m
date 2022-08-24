% code to generate the burgers dataset, obtained from https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/darcy
% number of realizations to generate
N = 1200; % use 1000 train, 200 test

%Parameters of covariance C = tau^(2*alpha-2)*(-Laplacian + tau^2 I)^(-alpha)
%Note that we need alpha > d/2 (here d= 2) 
%Laplacian has zero Neumann boundry
%alpha and tau control smoothness; the bigger they are, the smoother the
%function
alpha = 2;
tau = 3;

% grid size
s = 421;

input = zeros(N, s, s);
output = zeros(N, s, s);

for j=1:N
    %Generate random coefficients from N(0,C)
    norm_a = GRF(alpha, tau, s);
    
    %Another way to achieve ellipticity is to threshhold the coefficients
    thresh_a = zeros(s,s);
    thresh_a(norm_a >= 0) = 12;
    thresh_a(norm_a < 0) = 4;
    input(j,:,:) = thresh_a;
    
    %Forcing function, f(x) = 1 
    f = ones(s,s);

    %Solve PDE: - div(a(x)*grad(p(x))) = f(x)
    thresh_p = solve_gwf(thresh_a,f);
    output(j,:,:) = thresh_p;
    
    disp(j);
end

% end of loop, save the variables
filename = append('darcy_data_a',string(alpha),'_t',string(tau),'.mat');
coeff = input;
sol = output;
save(filename, 'coeff', 'sol', '-v7.3');
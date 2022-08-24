% code to generate the burgers dataset, obtained from https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/burgers
% modified by Yong Zheng Ong to include saving of matrix

% number of realizations to generate
N = 12000; % use 10000 train, 2000 test

% parameters for the Gaussian random field
% the higher the gamma, tau, the smoother the GRF
% note that we need alpha > d/2 (here d= 1) 
gamma = 2.5;
tau = 5;
sigma = tau^(2);

% viscosity
% edit this value to generate different viscosities for training
visc = 0.1;
% grid size
s = 1024;
steps = 200; % in the application, we learn from t=0 to t=1

input = zeros(N, s);
output = zeros(N, s);

tspan = linspace(0,1,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    u0 = GRF(s/2, 0, gamma, tau, sigma, "periodic");
    
    tic;
    u = burgers(u0, tspan, s, visc);
    
    u0eval = u0(x);
    toc;
    
    input(j,:) = u0eval(1:end-1);
    
    if steps == 1
        output(j,:) = u.values;
    else
        output(j,:) = u{steps+1}.values;
    end
    
    disp(j);
end

% end of loop, save the variables
filename = append('burgers_data_g',string(gamma),'_t',string(tau),'_v',string(visc),'.mat');
a = input;
u = output;
save(filename, 'a', 'u');


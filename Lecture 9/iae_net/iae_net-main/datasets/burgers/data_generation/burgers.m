% code to generate the burgers dataset, obtained from https://github.com/zongyi-li/fourier_neural_operator/tree/master/data_generation/burgers
function u = burgers(init, tspan, s, visc)

S = spinop([0 1], tspan);
dt = tspan(2) - tspan(1);
S.lin = @(u) + visc*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2);
S.init = init;
u = spin(S,s,dt,'plot','off'); 


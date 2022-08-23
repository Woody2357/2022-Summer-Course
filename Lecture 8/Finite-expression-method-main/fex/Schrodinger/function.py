import numpy as np
import torch
from torch import sin, cos, exp
import math

c = 3

def LHS_pde(u, x, dim_set):

    v = torch.ones(u.shape).cuda()
    bs = x.size(0)
    ux = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True)[0]
    uxx = torch.zeros(bs, dim_set).cuda()
    for i in range(dim_set):
        ux_tem = ux[:, i].reshape([x.size()[0], 1])
        uxx_tem = torch.autograd.grad(ux_tem, x, grad_outputs=v, create_graph=True)[0]
        uxx[:, i] = uxx_tem[:, i]

    LHS = -torch.sum(uxx, dim=1, keepdim=True)
    V = -exp(2/dim_set*torch.sum(cos(x), dim=1, keepdim=True))/(c**2)+torch.sum(sin(x)**2/(dim_set**2), dim=1, keepdim=True)-torch.sum(cos(x)/(dim_set), dim=1, keepdim=True)
    return LHS+u**3+V*u#ux+uy#uxx+uyy

# def RHS_pde(x):
#     # return -torch.sin(x)
#     # return 2*torch.cos(x)-x*torch.sin(x)
#     return torch.sin(x)# 2*torch.cos(x**2)-4*x**2*torch.sin(x**2) #torch.sin(x)
#
# def true_solution(x):
#     return x-torch.sin(x)#torch.sin(x**2)#torch.sin(x**2)#torch.sin(x)+x # -1*x+0*x**2#

def RHS_pde(x):
    # return 3 * torch.cos(2 * x[:, 0:1] + x[:, 1:2])
    # return 3 * torch.cos(2 * x[:, 0:1] + x[:, 1:2]) + 3*x[:, 0:1]
    # return 3 * torch.exp(2 * x[:, 0:1] + x[:, 1:2])
    # dim = x.size(1)
    bs = x.size(0)
    return torch.zeros(bs, 1).cuda()
    # dim = x.size(1)
    # coefficient = 2 * math.pi * torch.ones(1, dim).cuda()
    # coefficient[:, 0] = 1
    # print(x.size(), coefficient.size(), coefficient)
    # return -dim*math.pi*torch.cos(torch.sum(x * coefficient, dim=1, keepdim=True))

def true_solution(x):
    # return torch.sin(2*x[:,0:1]+x[:,1:2])
    # return torch.sin(2 * x[:, 0:1] + x[:, 1:2]) + 1.5*x[:, 0:1]**2
    dim = x.size(1)
    # coefficient[:,0] = 1
    # print(x.size(), coefficient.size(), coefficient)
    return exp(1/dim*torch.sum(cos(x), dim=1, keepdim=True))/(c)



# def RHS_pde(x):
#     # return -torch.sin(x)
#     # return 2*torch.cos(x)-x*torch.sin(x)
#     return torch.sin(x)+2# 2*torch.cos(x**2)-4*x**2*torch.sin(x**2) #torch.sin(x)
#
# def true_solution(x):
#     return x-torch.sin(x)+x**2

# unary_functions = [lambda x: 0*x**2,
#                    lambda x: 1+0*x**2,
#                    # lambda x: 5+0*x**2,
#                    lambda x: x+0*x**2,
#                    # lambda x: -x+0*x**2,
#                    lambda x: x**2,
#                    lambda x: x**3,
#                    lambda x: x**4,
#                    # lambda x: x**5,
#                    torch.exp,
#                    torch.sin,
#                    torch.cos,]
#                    # torch.erf,
#                    # lambda x: torch.exp(-x**2/2)]

unary_functions = [lambda x: 0*x**2,
                   lambda x: 1+0*x**2,
                   lambda x: x+0*x**2,
                   lambda x: x**2,
                   lambda x: x**3,
                   lambda x: x**4,
                   torch.exp,
                   torch.sin,
                   torch.cos,]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]


unary_functions_str = ['({}*(0)+{})',
                       '({}*(1)+{})',
                       # '5',
                       '({}*{}+{})',
                       # '-{}',
                       '({}*({})**2+{})',
                       '({}*({})**3+{})',
                       '({}*({})**4+{})',
                       # '({})**5',
                       '({}*exp({})+{})',
                       '({}*sin({})+{})',
                       '({}*cos({})+{})',]
                       # 'ref({})',
                       # 'exp(-({})**2/2)']

unary_functions_str_leaf= ['(0)',
                           '(1)',
                           # '5',
                           '({})',
                           # '-{}',
                           '(({})**2)',
                           '(({})**3)',
                           '(({})**4)',
                           # '({})**5',
                           '(exp({}))',
                           '(sin({}))',
                           '(cos({}))',]


binary_functions_str = ['(({})+({}))',
                        '(({})*({}))',
                        '(({})-({}))']

if __name__ == '__main__':
    batch_size = 200
    left = -1
    right = 1
    points = (torch.rand(batch_size, 1)) * (right - left) + left
    x = torch.autograd.Variable(points.cuda(), requires_grad=True)
    function = true_solution

    '''
    PDE loss
    '''
    LHS = LHS_pde(function(x), x)
    RHS = RHS_pde(x)
    pde_loss = torch.nn.functional.mse_loss(LHS, RHS)

    '''
    boundary loss
    '''
    bc_points = torch.FloatTensor([[left], [right]]).cuda()
    bc_value = true_solution(bc_points)
    bd_loss = torch.nn.functional.mse_loss(function(bc_points), bc_value)

    print('pde loss: {} -- boundary loss: {}'.format(pde_loss.item(), bd_loss.item()))
import numpy as np

# H = np.genfromtxt('h_pred.txt')
# E = np.genfromtxt('e_pred.txt')
S = np.genfromtxt('s_predt.txt')
HX = np.genfromtxt('hx_predt.txt')
C = np.genfromtxt('c_predt.txt')



C_sum = np.sum(np.absolute(np.round(C,4)))
HX_sum = np.sum(np.absolute(np.round(HX,4)))
S_sum = np.sum(np.absolute(np.round(S,4)))


print (C_sum + HX_sum)
print (S_sum)
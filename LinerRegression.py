from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
xs=np.array([1,2,3,4,5,6],dtype=np.float64) # define data type as float 64
ys=np.array([5,4,7,8,9,10],dtype=np.float64)

def best_fit_slope_and_intercept(xset,yset):
    m=( ( (mean(xset)*mean(yset))-mean(xset*yset) ) /
       ( (mean(xset)**2) -mean(xset**2)) )
    b=(mean(yset)-m*mean(xset))
    return m,b

def squared_error(ys_orig,y_line):
    return sum((y_line-ys_orig)**2)

def coeffecient_of_determination(ys_orig,y_line):
    y_meanline=[mean(ys_orig) for y in ys_orig]
    sq_err_reg_line=squared_error(ys_orig,y_line)
    sq_err_ymean_line=squared_error(ys_orig,y_meanline)
    return 1-(sq_err_reg_line/sq_err_ymean_line)

m,b=best_fit_slope_and_intercept(xs,ys)
regression_line=[(m*x+b) for x in xs]
# Coeffecient of determination...how good is your best fit determined by r Squqred theory
r_squared=coeffecient_of_determination(ys,regression_line)
print(r_squared)
plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()
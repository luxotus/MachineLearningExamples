import numpy
import scipy.optimize as optimization

# Generate artificial data = straight line with a=0 and b=1 plus some noise.
xData = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
yData = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])

# Initial guess.
x0 = numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# The function whose square is to be minimized.
# params ... list of parameters tuned to minimise function.
# Further arguments:
# xData ... design matrix for a linear model.
# yData ... observed data.
def model(params, xs, ys):
  return (ys - numpy.dot(xs, params))

def func(x, a, b):
  return a + b*x

print(optimization.leastsq(model, x0, args=(xData, yData)))
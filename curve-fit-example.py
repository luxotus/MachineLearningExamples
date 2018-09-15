import numpy
import scipy.optimize as optimization

# Generate artificial data = straight line with a=0 and b=1
# plus some noise.
xdata = numpy.array([0.0,1.0,2.0,3.0,4.0,5.0])
ydata = numpy.array([0.1,0.9,2.2,2.8,3.9,5.1])

# Initial guess.
x0    = numpy.array([0.0, 0.0, 0.0])

sigma = numpy.array([1.0,1.0,1.0,1.0,1.0,1.0])

def model(x, a, b, c):
  return a + b*x + c*x*x

print(optimization.curve_fit(model, xdata, ydata, x0, sigma))
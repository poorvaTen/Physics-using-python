# Physics-using-python

#Addition and Multiplication of vectors using python.
 x = np.array([3,2])
 y = np.array([5,1])
 z = x + y


 #scalar product of two vectors.
 x = np.array([3,2])
 y = np.array([5,1])
 z = x + y

x = np.array([1,2,3])
y = np.array([2,4,6])
np.dot(x,y)
28
dot = np.dot(x,y)
x_modulus = np.sqrt((x*x).sum())
y_modulus = np.sqrt((y*y).sum())
cos_angle = dot / x_modulus / y_modulus # cosine of angle between x and y
angle = np.arccos(cos_angle)
angle
0.80823378901082499
angle * 360 / 2 / np.pi # angle in degrees
46.308384970187326



#Cross Product of two vectors .
x = np.array([1,1,1])
y = np.array([2,5,6])

np.cross(x,y)
array([1,  -4,  3])

np.cross(y,x)
array([-1, 4, 3])

#Gradient,Divergence and Curl.

#Calulating Gradient using numpy.
from numpy import *

x,y,z = mgrid[-50:100:35., -50:100:35., -50:100:35.]

V = 2*x**2 + 3*y**2 - 4*z # just a random function for the potential

Ex,Ey,Ez = gradient(V)



#Divergence of a vector field.

import numpy as np

def divergence(field):
       "return the divergence of a n-D field"
        return np.sum(np.gradient(field),axis=0)


#Curl of a vector.
#let there be F such that:
#F = (y2z,-xy,z2) = y2zx - xyy + z2z, then y would be R[1], x is R[0] and z is R[2]
#while the vectors of the 3 axes would be R.x, R.y, R.z

from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl
        R = ReferenceFrame('R')

        F = R[1]**2 * R[2] * R.x - R[0]*R[1] * R.y + R[2]**2 * R.z

        G = curl(F, R)

#Differential Equations :

# Solving 1st order differential equations:

from math import sin
from numpy import array,arange
from pylab import plot,xlabel,show


def f(r,t):
    x = r[0]
    y = r[1]
    fx = x*y - x
    fy = y - x*y + sin(t)**2
    return array([fx,fy],float)

a = 0.0
b = 10.0
N = 1000
h = (b-a)/N

tpoints = arange(a,b,h)
xpoints = []
ypoints = []

r = array([1.0,1.0],float)
for t in tpoints:
    xpoints.append(r[0])
    ypoints.append(r[1])
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
plot(tpoints,xpoints)
plot(tpoints,ypoints)
xlabel("t")
show()


#Second order differential equation:

from math import sin
from numpy import arange
from pylab import plot,xlabel,ylabel,show

def f(x,t):
    return -x**3 + sin(t)

a = 0.0
b = 10.0
N = 10
h = (b-a)/N

tpoints = arange(a,b,h)
xpoints = []

x = 0.0
for t in tpoints:
    xpoints.append(x)
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1,t+0.5*h)
    x += k2

plot(tpoints,xpoints)
xlabel("t")
ylabel("x(t)")
show()






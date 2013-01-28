from pylab import *
from numpy import outer

x = outer(arange(0, 1, 0.01), ones(100))

print(x)

imshow(transpose(x), cmap=cm.hsv)
show()

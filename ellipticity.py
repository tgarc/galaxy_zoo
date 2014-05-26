#!/bin/env python

import numpy as np
import math

# ######## Hu's invariant moments, and eccentricity #########
# References:
#
# [1] Andry M. Pinto, Luis F. Rocha, and A. Paulo Moreira. "Object recognition using 
#     laser range finder and machine learning techniques." Robotics and Computer-Integrated 
#     Manufacturing 29.1 (2013): 12-22.

# Equation 1 [1]
# "F" should be a matrix containing single-channel image data
def mt_geometric_moment(p, q, F):
	m = len(F)
	n = len(F[0])
	sum = 0
	for i in range(m):
		for j in range(n):
			sum = sum + i**p*j**q*F[i][j]
	return sum
	
# Equation 2, part 1 [1]
def mt_x_centroid(F):
	return mt_geometric_moment(1, 0, F) / float(mt_geometric_moment(0, 0, F))

# Equation 2, part 2 [1]
def mt_y_centroid(F):
	return mt_geometric_moment(0, 1, F) / float(mt_geometric_moment(0, 0, F))
	
# Equation 3 [1]
def mt_central_geometric_moment(p, q, F):
	m = len(F)
	n = len(F[0])
	xb = mt_x_centroid(F)
	yb = mt_y_centroid(F)
	sum = 0
	for i in range(m):
		for j in range(n):
			sum = sum + (i-xb)**p*(j-yb)**q*F[i][j]
	return sum
	
# Equation 4 [1]
def mt_covariance_matrix(F):
	mass = mt_geometric_moment(0, 0, F)
	J = []
	J.append([mt_central_geometric_moment(2, 0, F) / float(mass), mt_central_geometric_moment(1, 1, F) / float(mass)])
	J.append([mt_central_geometric_moment(1, 1, F) / float(mass), mt_central_geometric_moment(0, 2, F) / float(mass)])
	return J
	
# Equation 5 [1]
def mt_rotation_angle(F):
	numerator = 2 * mt_central_geometric_moment(1, 1, F)
	denominator = mt_central_geometric_moment(2, 0, F) - mt_central_geometric_moment(0, 2, F)
	return 0.5 * math.atan2(numerator, denominator) # atan2 computes the correct quadrant
	
# Eccentricity calculation (section 3.1.2) [1]
def mt_eccentricity(F):
	J = mt_covariance_matrix(F)
	eigenvalues, eigenvectors = np.linalg.eig(J)
	ratio = eigenvalues[1] / eigenvalues[0]
	if ratio>1:
		# Just in case
		ratio = 1.0 / ratio
	return np.lib.scimath.sqrt(1 - ratio)
	
# Equation 6 [1]
# Normalized central geometric moment
def mt_ncgm(p, q, F):
	upq = mt_central_geometric_moment(p, q, F)
	denominator = mt_central_geometric_moment(0, 0, F)**(((p+q)/2.0)+1)
	return upq / float(denominator)
	
# Hu's invariant moments, I1 - I8 (section 3.1.4) [1]
def mt_I1(F):
	return mt_ncgm(2, 0, F) + mt_ncgm(0, 2, F)
	
def mt_I2(F):
	return (mt_ncgm(2, 0, F) - mt_ncgm(0, 2, F))**2 + 4*mt_ncgm(1, 1, F)
	
def mt_I3(F):
	return (mt_ncgm(3, 0, F) - 3*mt_ncgm(1, 2, F))**2 + (3*mt_ncgm(2, 1, F) - mt_ncgm(0, 3, F))**2
	
def mt_I4(F):
	return (mt_ncgm(3, 0, F) + mt_ncgm(1, 2, F))**2 + (mt_ncgm(2, 1, F) + mt_ncgm(0, 3, F))**2
	
def mt_I5(F):
	n30 = mt_ncgm(3, 0, F)
	n12 = mt_ncgm(1, 2, F)
	n03 = mt_ncgm(0, 3, F)
	n21 = mt_ncgm(2, 1, F)
	return (n30-3*n12)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2) + (3*n21-n03)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)
	
def mt_I6(F):
	n30 = mt_ncgm(3, 0, F)
	n12 = mt_ncgm(1, 2, F)
	n03 = mt_ncgm(0, 3, F)
	n21 = mt_ncgm(2, 1, F)
	n20 = mt_ncgm(2, 0, F)
	n02 = mt_ncgm(0, 2, F)
	n11 = mt_ncgm(1, 1, F)
	return (n20-n02)*((n30+n12)**2-(n21+n03)**2) + 4*n11*(n30+n12)*(n21+n03)

def mt_I7(F):
	n30 = mt_ncgm(3, 0, F)
	n12 = mt_ncgm(1, 2, F)
	n03 = mt_ncgm(0, 3, F)
	n21 = mt_ncgm(2, 1, F)
	return (3*n21-n03)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2) - (n30-3*n12)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)
	
def mt_I8(F):
	n30 = mt_ncgm(3, 0, F)
	n12 = mt_ncgm(1, 2, F)
	n03 = mt_ncgm(0, 3, F)
	n21 = mt_ncgm(2, 1, F)
	n20 = mt_ncgm(2, 0, F)
	n02 = mt_ncgm(0, 2, F)
	n11 = mt_ncgm(1, 1, F)
	return n11*((n30+n12)**2-(n03+n21)**2) - (n20-n02)*(n30+n12)*(n03+n21)
	
# Hu's invariant moments, except that the necessary normalized central geometric moments
# are passed as parameters. This saves computation time when computing multiple invariant
# moments. The "p" stands for parallel
def mt_I1_p(n20, n02):
	return n20 + n02
	
def mt_I2_p(n20, n02, n11):
	return (n20 - n02)**2 + 4*n11
	
def mt_I3_p(n30, n12, n21, n03):
	return (n30 - 3*n12)**2 + (3*n21 - n03)**2
	
def mt_I4_p(n30, n12, n21, n03):
	return (n30 + n12)**2 + (n21 + n03)**2
	
def mt_I5_p(n30, n12, n21, n03):
	return (n30-3*n12)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2) + (3*n21-n03)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)
	
def mt_I6_p(n20, n02, n30, n12, n21, n03, n11):
	return (n20-n02)*((n30+n12)**2-(n21+n03)**2) + 4*n11*(n30+n12)*(n21+n03)

def mt_I7_p(n30, n12, n21, n03):
	return (3*n21-n03)*(n30+n12)*((n30+n12)**2-3*(n21+n03)**2) - (n30-3*n12)*(n21+n03)*(3*(n30+n12)**2-(n21+n03)**2)
	
def mt_I8_p(n20, n02, n30, n12, n21, n03, n11):
	return n11*((n30+n12)**2-(n03+n21)**2) - (n20-n02)*(n30+n12)*(n03+n21)

# A function that displays the invariant moments of an image, and other statistics.
def mt_display_moments(F):
	print "I1:", mt_I1(F)
	print "I2:", mt_I2(F)
	print "I3:", mt_I3(F)
	print "I4:", mt_I4(F)
	print "I5:", mt_I5(F)
	print "I6:", mt_I6(F)
	print "I7:", mt_I7(F)
	print "I8:", mt_I8(F)
	print "Area:", mt_area(F)
	print "Eccentricity:", mt_eccentricity(F)
	print "Rotation:", mt_rotation_angle(F)
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################
# Initialising tensors #
########################

x = tf.constant(4.0, shape=(1, 1))
print(x)

# Makes a 2x3 matrix
x = tf.constant([[1, 2, 3], [4, 5, 6]])
print(x)

# Makes a 3x3 matrix filed with 1s
x = tf.ones((3, 3))
print(x)

# Makes identity matrix for given size
x = tf.eye(3)
print(x)

# Makes a 3x3 where values are randomised from a normal distribution
x = tf.random.normal((3, 3), mean=0, stddev=1)
print(x)

# Makes a vector of the numbers from 0 to n-1
x = tf.range(9)
print(x)

# Makes a vector of the numbers 2 apart starting with 0 and ending at 10 (exclusive of the final number)
x = tf.range(start=0, limit=10, delta=2)
print(x)
# Casts the values of X to different data types
x = tf.cast(x, dtype=tf.float64)
print(x)

###########################
# Mathematical Operations #
###########################

# Working with 2 vectors
vect1 = tf.constant([1, 2, 3])
vect2 = tf.constant([9, 8, 7])

# Simple elementwise operations on 2 vectors
ans = tf.add(vect1, vect2)  # This is the same as "ans = vect1 + vect2"
print(ans)
ans = tf.subtract(vect1, vect2)  # This is the same as "ans = vect1 - vect2"
print(ans)
ans = tf.divide(vect1, vect2)  # This is the same as "ans = vect1 / vect2"
print(ans)
ans = tf.multiply(vect1, vect2)  # This is the same as "ans = vect1 * vect2"
print(ans)

# Finding the dot-product of 2 vectors
ans = tf.tensordot(vect1, vect2, axes=1)  # Can also be done using "ans = tf.reduce_sum(vect1 * vect2, axis=0)"
print(ans)

# Exponentiation on each element
ans = vect1 ** 5  # This acts as if the vector is a normal number
print(ans)

# Matrix multiplication
mat1 = tf.random.normal((2, 3))
mat2 = tf.random.normal((3, 4))
ans = tf.matmul(mat1, mat2)  # This is the same as "ans = mat1 @ mat2"
print(ans)

# Finds the determinant of a 3x3 matrix
mat3 = tf.constant([[1., 5., 11.],
                    [3., 6., 300.],
                    [4., 10., 14.]])
mat3 = tf.cast(mat3,
               dtype=tf.float64)  # Have to cast to 64bit float as an overflow error occurred when using 32bit float
det = tf.linalg.det(mat3, name=None)
print(mat3)
print(det)

#####################
# Indexing a Tensor #          # Acts very much like a normal list in Python
#####################

x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])      # Prints all elements in the tensor
print(x[1:])      # Prints all elements from index value 1 to the end
print(x[1:3])      # Prints all elements from index value 1 (inclusive) to index value 3 (exclusive)
print(x[::2])      # Prints every other element, e.g. indexes 0, 2, 4, 6, ...
print(x[::-1])      # Prints all elements in reverse order

# Prints the elements that have the index values listed
indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)
print(x_ind)

# Printing elements from an array
x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
print(x[0])      # Prints the first row of the matrix
print(x[0, 0])      # Prints the first element of the first row of the matrix
print(x[0:2, :])      # Prints all the elements of the first two rows of the matrix

#############
# Reshaping #
#############

x = tf.range(9)
print(x)
# Reshapes the 9 elements (0-8) into a 3x3 matrix
x = tf.reshape(x, (3, 3))
print(x)
# Changes the orientation of the matrix so [0, 1, 2] becomes the first column instead of the first row
x = tf.transpose(x, perm=[1, 0])
print(x)
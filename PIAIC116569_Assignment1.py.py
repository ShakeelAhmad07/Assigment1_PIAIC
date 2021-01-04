#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


arr = np.zeros(10)
arr


# 3. Create a vector with values ranging from 10 to 49

# In[10]:


vc = np.arange(10, 50)
print("values ranging from 10 to 49: ", vc)
print(vc.dtype)


# 4. Find the shape of previous array in question 3

# In[8]:


print(vc.shape)


# 5. Print the type of the previous array in question 3

# In[9]:


print(vc.dtype)


# 6. Print the numpy version and the configuration
# 

# In[12]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[15]:


print("Print the dimension of the array in question 3: ", vc.ndim)


# 8. Create a boolean array with all the True values

# In[19]:


boo = np.ones(10, dtype=bool)
print(boo)


# 9. Create a two dimensional array
# 
# 
# 

# In[20]:


d2_arr = np.array([
                   [1,2,3,4,5],
                   [6,7,8,9,10]
    
                  ])
print(d2_arr, d2_arr.ndim)


# 10. Create a three dimensional array
# 
# 

# In[24]:


d3_arr = np.array([
                   [
                       [1,2,3,4,5]
                   ],
                   [
                       [6,7,8,9,10]
                   ],
                   [
                       [16,17,18,19,110]
                   ]
    
    
                  ])
print(d3_arr.ndim,d3_arr.shape)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[30]:


vv = np.arange(10)
print(vv)
vv = vv[::-9]
print(vv)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[36]:


vc = np.zeros(10)
print(vc)
vc[4] =1
print(vc)


# 13. Create a 3x3 identity matrix

# In[38]:


ii = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]
              ])
print(ii)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[45]:


arr = np.array([1,2,3,4,5] ,dtype =np.float)
print(" Array int0 float: ",arr)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[46]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
ml = arr1*arr2
print(ml)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[106]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
print(arr1,arr2)
comp = arr1 == arr2
print(comp)


# 17. Extract all odd numbers from arr with values(0-9)

# In[52]:


old = np.arange(10)
print(old)
xx = np.where(old%2== 0)
print(xx)


# 18. Replace all odd numbers to -1 from previous array

# In[104]:


old = np.arange(1,10)
print(old)
xx = np.where(old%2== 0)
print(xx)
xx[xx]=-1


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[6]:


arr = np.arange(10)
print(arr)
arr[5] = 12
arr[6] = 12
arr[7] = 12
arr[8] = 12
print("Replace the values of indexes 5,6,7 and 8 to 12: " ,arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[16]:


xx = np.ones((6,6))
print("Original array:")
print(xx)
xx[1: -1,1: -1] =0
print("Array with 1 on the border and 0 inside")
print(xx)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[18]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
print(arr2d)
arr2d[1,1] = 12
print("Replace the value 5 to 12: ", arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[22]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
arr3d[0] =64
print("Convert all the values of 1st array to 64: ",arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[65]:


arr3d = np.arange(0,9).reshape(3,3)
arr3d
#arr3dd = arr3d[:1]
#print(arr3dd)


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[95]:


de = np.arange(2,10).reshape(2,4)
print(de, de.ndim)
de[:1,:1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[93]:


de = np.arange(2,10).reshape(2,4)
print(de)
de[:1]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[78]:


aa = np.random.random((10,10))
print("Original Array:")
print(aa) 
xmin, xmax = aa.min(), aa.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[87]:



a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("A: ",a )
print("B: ",b )
print("Common values between two arrays:")
print(np.intersect1d(a, b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[89]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("A: ",a )
print("B: ",b )
print("The positions where elements of a and b match")
print(np.searchsorted(a, np.intersect1d(a, b)))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[12]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)

d = data[(names != 'Will')]
print(d)


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[13]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
ee = data[(names != 'Will') & (names != 'Joe')]
print(ee)


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[20]:


d2 = np.arange(15).reshape(5,3)
print(d2)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[19]:


ddd = np.arange(16).reshape(2,2,4)
print(ddd)


# 33. Swap axes of the array you created in Question 32

# In[55]:


ddd = np.arange(16).reshape(2,2,4)
print(ddd)
np.swapaxes(ddd,1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[26]:


ss = np.arange(10)
print(ss)
np.sqrt(ss)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[40]:


aa = np.random.random(12)
bb = np.random.random(12)
print("AA : ",aa)

print("BB: ",bb)
np.maximum(aa, bb)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[47]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
qq=(np.unique(names))
print(qq)
sorted(set(qq))


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[46]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
print("AA : ",a)

print("BB: ",b)
res = np.setdiff1d(a,b)
print(res)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[76]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 
print(sampleArray)
de = np.delete(sampleArray, 2,0)
print(de)
ewColumn = np.array([[10],[10],[10]])
an_array = np.append(sampleArray, ewColumn, axis=1)
print("Insert New Column: ",an_array)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[51]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(x,y)
np.dot(x, y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[57]:


g = np.random.random(20)
print(g)
np.cumsum(g, axis=0, out=g)


# In[ ]:





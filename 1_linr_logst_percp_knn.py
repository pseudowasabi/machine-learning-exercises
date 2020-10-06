# only for code snippets
# subjects: linear regression, perceptron, logistic regression, kNN
# numpy, operator, sklearn should be imported

# 1. calculating cost function for linear regression

# print("below is cost_naive function")
# print(X.shape)
# print(theta.shape)
for i in range(n):
    hypothesis = theta[0] * X[i][0] + theta[1] * X[i][1]
    J += (y[i] - hypothesis) ** 2
    # print(i, hypothesis, J)
J /= (2 * n)
# print(J)
    

# 2. calculating cost function for linear reg, using matrix operation
hypothesis = X.dot(theta)
J = sum((y - hypothesis) ** 2) / (2 * n)


# 3. getting OLS (Ordinary Least Squares, a closed form solution)

XT = np.transpose(X)
theta = np.linalg.inv(XT.dot(X)).dot(XT).dot(y)


# 4. Gradient descent (naive)

cost_sum_mul_x_tilda = np.zeros(2)
for j in range(n):
    hypothesis = theta[0] * X[j][0] + theta[1] * X[j][1]
    '''
    cost_sum_mul_x_tilda[0] += (y[j] - hypothesis) * X[j][0]
    cost_sum_mul_x_tilda[1] += (y[j] - hypothesis) * X[j][1]
    '''
    # 위의 식을 아래의 식 하나로 작성 가능 (vector에 대해서도 바로 연산)
    cost_sum_mul_x_tilda += (y[j] - hypothesis) * X[j]

gradient_of_J = - (1 / n) * cost_sum_mul_x_tilda
theta = theta - alpha * gradient_of_J

# print(cost_naive(X, y, theta))


# 5. Gradient descent (using matrix operation)
        
hypothesis = X.dot(theta)
cost_sum_mul_x_tilda = np.transpose(y - hypothesis).dot(X)

gradient_of_J = -(1 / n) * cost_sum_mul_x_tilda
theta = theta - alpha * gradient_of_J
        

# 6. SGD (Stochastic GD)

indices = np.arange(0, n)
random_index = np.random.choice(indices, mini_batch, replace=True)
X_I = X[random_index[0]]
y_I = y[random_index[0]]

hypothesis = theta[0] * X_I[0] + theta[1] * X_I[1]
stochastic_gradient = (y_I - hypothesis) * X_I

theta = theta + alpha * stochastic_gradient
           

# 7. Perceptron

theta_transpose_x_tilda = X.dot(theta)
preds = np.array([1. if theta_transpose_x_tilda[i] > 0 else 0. for i in range(len(X))])


# 8. Cost function in Perceptron 

# get hypothesis first
hypothesis = perceptron(X, theta)
#theta_transpose_x_tilda = X.dot(theta)

'''
hypothesis = np.zeros(n)
for i in range(n):
    hypothesis[i] = 1
    if theta_transpose_x_tilda[i] > 0:
        hypothesis[i] = 1
    else:
        hypothesis[i] = 0
print(hypothesis)
'''

#hypothesis = np.array([1 if theta_transpose_x_tilda[i] > 0 else 0 for i in range(n)])
#print(hypothesis)

# next, calculate average of loss function
J = (-1 / n) * (y - hypothesis).dot(X.dot(theta))


# 9. Updating perceptron (like GD)

hypothesis = perceptron(X, theta)
J2 = (1 / n) * (y - hypothesis).dot(X)
theta = theta + alpha * J2


# 10. Sigmoid function

n = len(x)
#print("sigmoid!")
#print(x)
sig_x = np.array([1 / (1 + np.exp(-x[i])) for i in range(n)])
#print(sig_x)


# 11. Cost function in logistic regression

hypothesis = sigmoid(X.dot(theta))
for i in range(n):
    J += (y[i] * np.log(hypothesis[i]) + (1 - y[i]) * np.log(1 - hypothesis[i]))
J *= (-1 / n)
    

# 12. GD in logistic regression

hypothesis = sigmoid(X.dot(theta))
for i in range(n): # res = shape (,3)
    grad += (y[i] - hypothesis[i]) * X[i]
#print(grad)
grad *= (-1 / n)
#grad = grad.tolist() # list로 보내면 logistic_optimal_theta를 구하는 과정에서 오류 발생


# 13. Prediction in logistic-reg

hypothesis = sigmoid(X.dot(theta))
preds = np.array([1. if hypothesis[i] >= 0.5 else 0. for i in range(n)])


# 14. get Euclidean distance for kNN
    
for i in range(m):
    distances[i] = np.sqrt(np.square(targetX[0] - dataSet[i][0]) + np.square(targetX[1] - dataSet[i][1]))
#print(distances)
    

# 15. getting k-Nearest Neighbor

distances = euclideanDistance(targetX, dataSet)
sorted_idx = distances.argsort()
for i in range(k):
    #closest_data[i] = sorted_idx[i+1]
    closest_data[i] = sorted_idx[i] # 자기 자신 포함해서 넣기
#print(closest_data)
    

# 16. Prediction using kNN

closest_data = getKNN(targetX[i], dataSet, labels, k)
#print(closest_data)
temporary_sum = 0
for j in range(k):
    #print(closest_data[j])
    if labels[int(closest_data[j])] == 1:
        temporary_sum += 1
temporary_sum /= k

if temporary_sum > 0.5:
    predicted_array[i] = 1
elif temporary_sum < 0.5:
    predicted_array[i] = 0
else: # ties
    predicted_array[i] = labels[int(closest_data[0])]
        


# end. (to execute above code snippets, some parameter X, y, and iteration process wraping above code should be added)

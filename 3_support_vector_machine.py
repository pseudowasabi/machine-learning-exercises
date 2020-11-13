# only for code snippets
# subjects: several SVMs -> non-seperable (linear), non-linear (using kernel), grid search
# numpy, sklearn should be imported

### 1. non-seperable (linear) SVM

## 1-1. score function of svm
# theta_transpose_x_tilda = np.insert(X, 0, 1, axis=1).dot(theta)

n = X.shape[0]
d = X.shape[1] + 1

theta_transpose_x_tilda = np.zeros((n))
for i in range(n):
    theta_transpose_x_tilda[i] = theta[0]
    for j in range(1, d):
        theta_transpose_x_tilda[i] += (theta[j] * X[i][j - 1])

score = np.multiply(y, theta_transpose_x_tilda)


## 1-2. prediction function of svm
# n = X.shape[0]
# theta_transpose_x_tilda = np.insert(X, 0, 1, axis=1).dot(theta)
# print(theta_transpose_x_tilda)

n = X.shape[0]
d = X.shape[1] + 1

theta_transpose_x_tilda = np.zeros((n))
for i in range(n):
    theta_transpose_x_tilda[i] = theta[0]
    for j in range(1, d):
        theta_transpose_x_tilda[i] += (theta[j] * X[i][j - 1])

for i in range(n):
    if theta_transpose_x_tilda[i] > 0:
        prediction[i] = 1
    elif theta_transpose_x_tilda[i] < 0:
        prediction[i] = -1
    else:
        prediction[i] = 0


## 1-3. hinge loss
score = score_function(X, y, theta)
n = X.shape[0]

for i in range(n):
    if (1 - score[i]) >= 0:
        loss[i] = (1 - score[i])
    else:
        loss[i] = 0


## 1-4. objective function of svm
l2_norm_of_theta_1 = 0
d = theta.shape[0]

for i in range(1, d):
    l2_norm_of_theta_1 += (theta[i] ** 2)
half_l2_norm = (1 / 2) * l2_norm_of_theta_1

h_loss = hinge_loss(X, y, theta)
obj = half_l2_norm + C * sum(h_loss)


## 1-5. update svm theta parameters - perform below code in iteration
# reference - https://towardsdatascience.com/solving-svm-stochastic-gradient-descent-and-hinge-loss-8e8b4dd91f5b
n = X.shape[0]
d = X.shape[1] + 1
grad = np.zeros((d))

h_loss = hinge_loss(X, y, updated_theta)  # shape: (# of sample,)

for j in range(d):
    if j > 0:  # theta 1
        sum_of_y_mul_x = 0
        # calculate gradient of hinge loss for each theta coeff. (j)
        for k in range(n):  # vertical sum (for each j, calc all samples)
            if h_loss[k] >= 0:
                sum_of_y_mul_x += (y[k] * X[k][j - 1])
                # gradient 0 when loss is negative
        grad[j] = updated_theta[j] - C * sum_of_y_mul_x
    # else:       # theta 0
    # grad[j] = 0 - C * sum(y)

updated_theta = updated_theta - alpha * grad



### 2. non-linear (kernel-using) SVM and grid search

## 2-1. get polynomial kernel
poly = x.dot(z) + bias
K = 1.
for i in range(degree):
    K *= poly

## 2-2. get gaussian kernel
x_minus_z = x - z
K = np.exp(-x_minus_z.dot(x_minus_z) / (2 * (sigma ** 2)))

## 2-3. perform grid search using GridSearchCV in scikit-learn module
svc = SVC(kernel='rbf')
clf = GridSearchCV(estimator=svc, param_grid=parameters, cv=10)
clf.fit(X, y)
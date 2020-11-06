# only for code snippets
# subjects: regularization(ridge, lasso, l2-logistic), k-fold cross-validation, confusion matrix, ROC curve
# numpy, operator, sklearn should be imported


### 1. Ridge regression

## 1-1. cost method
# y = y.values # Pandas series to Numpy array
# cost function from linear regression
summation = 0
# y_pandas_series_to_list = y.to_list()

_hypo = X.dot(theta1)
for i in range(n):
    hypo = _hypo[i] + theta0
    # summation += ((y_pandas_series_to_list[i] - hypo) ** 2)
    # summation += ((y[y.index[i]] - hypo) ** 2)
    summation += ((y[i] - hypo) ** 2)
summation /= n

# penalty term - l2 norm square for Ridge regression
l2_norm_sqr = 0
theta_dim = X.shape[1]
for i in range(theta_dim):
    l2_norm_sqr += (theta1[i] ** 2)

J = summation + lamb * l2_norm_sqr

## 1-2. update method (should perform below codes with iteration)
J_all[i] = self._cost(X, y, theta0, theta1, lamb)
# using gradient of cost function in ridge regression

# for theta 1
hypo_1 = X.dot(theta1)
y_minus_hypo = np.zeros(n)

for j in range(n):
    # y_minus_hypo[j] = y[y.index[j]] - (hypo_1[j] + theta0)
    y_minus_hypo[j] = y[j] - (hypo_1[j] + theta0)
# print("Here is _update")
cost_sum_mul_x = y_minus_hypo.dot(X)  # same shape as theta1

'''
l2_norm = 0
theta_dim = X.shape[1]
for j in range(theta_dim):
    l2_norm += (theta1[i] ** 2)
l2_norm = np.sqrt(l2_norm)
print("l2norm", l2_norm)'''

'''
gradient_of_J_1 = -(1 / n) * cost_sum_mul_x + 2 * lamb * theta1
theta1 = theta1 - alpha * gradient_of_J_1'''

gradient_of_J_1 = -(2 / n) * cost_sum_mul_x + 2 * lamb * theta1
theta1 = theta1 - (alpha * gradient_of_J_1)

# for theta 0
'''
hypo_0 = theta0
gradient_of_J_0 = (-1 / n) * sum(cost_sum_mul_x) + 2 * lamb * sum(theta1)
print(gradient_of_J_0)
theta0 -= alpha * gradient_of_J_0
'''

hypo_0 = 0
for i in range(n):
    # hypo_0 += (y[y.index[i]] - theta0)
    hypo_0 += (y[i] - theta0) * 1  # since last term of x_tilda is 1.
gradient_of_J_0 = -(2 / n) * hypo_0
theta0 = theta0 - (alpha * gradient_of_J_0)

## 1-3. fit method
# X => Numpy array, y => Pandas series
if str(type(y)) == "<class 'pandas.core.series.Series'>":
    y = y.values  # Pandas series to Numpy array

self.theta0, self.theta1, self.J_all = self._update(X, y, self.theta0, self.theta1, self.num_iters, self.alpha, self.lamb)

## 1-4. predict method
pred = X.dot(self.theta1) + self.theta0

## 1-5. closed form of ridge regrssion
y = y.values  # Pandas series to Numpy array

summation = 0
for i in range(n):
    # summation += y[y.index[i]]
    summation += y[i]
theta_0 = summation / n

XT = np.transpose(X)
theta_1 = np.linalg.inv((1 / n) * XT.dot(X) + lamb * np.identity(X.shape[1])).dot(XT).dot(y) * (1 / n)

## 1-6. closed form ridge regrssion predict method
pred = X.dot(theta_1) + theta_0


### 2. L2-logistic regression

## 2-1. cost method
summation = 0

X_dot_theta = X.dot(theta1)
for i in range(n):
    X_dot_theta[i] += theta0
hypothesis = self.sigmoid(X_dot_theta)

for i in range(n):
    summation += (y[i] * np.log(hypothesis[i]) + (1 - y[i]) * np.log(1 - hypothesis[i]))
summation *= (-1 / n)

# penalty term - l2 norm square for Ridge regression
l2_norm_sqr = 0
theta_dim = X.shape[1]
for i in range(theta_dim):
    l2_norm_sqr += (theta1[i] ** 2)
J = summation + lamb * l2_norm_sqr

## 2-2. update method (should perform below codes with iteration)
# print("iters:", i, end=", ")
J_all[i] = self._cost(X, y, theta0, theta1, lamb)
# using gradient of cost function in norm-regularized logistic regression

# for theta 1
X_dot_theta = X.dot(theta1)
for i in range(n):
    X_dot_theta[i] += theta0
hypo = self.sigmoid(X_dot_theta)
'''
y_minus_hypo = np.zeros(n)

for j in range(n):
    y_minus_hypo[j] = y[j] - hypo[j]
'''
cost_sum_mul_x = (y - hypo).dot(X)  # same shape as theta1

gradient_of_J_1 = -(1 / n) * cost_sum_mul_x + 2 * lamb * theta1
theta1 = theta1 - (alpha * gradient_of_J_1)

# for theta 0
hypo_0 = 0
for i in range(n):
    hypo_0 += (y[i] - hypo[i]) * 1  # since last term of x_tilda is 1.
gradient_of_J_0 = -(1 / n) * hypo_0
theta0 = theta0 - (alpha * gradient_of_J_0)
# print("theta1:", theta1, ", theta0:", theta0)
# print("theta0:", theta0)

## 2-3. fit method
# X => Numpy array, y => Pandas series
if str(type(y)) == "<class 'pandas.core.series.Series'>":
    y = y.values  # Pandas series to Numpy array

self.theta0, self.theta1, self.J_all = self._update(X, y, self.theta0, self.theta1, self.num_iters, self.alpha, self.lamb)

## 2-4. predict method
for i in range(n):
    probs[i] = self.sigmoid(X[i].dot(self.theta1) + self.theta0)
    if probs[i] >= 0.5:
        preds[i] = 1
    else:
        preds[i] = 0


### 3. k-Fold Cross validation

## 3-1. get k-Fold
# print("fold size", fold_size)

for i in range(n_splits):
    train_idx = []
    val_idx = []

    for j in range(0, i * fold_size):
        # print(j, end = ' ')
        train_idx.append(indices[j])
    # print()
    if i != n_splits - 1:
        for j in range(i * fold_size, (i + 1) * fold_size):
            # print(j, end=' ')
            val_idx.append(indices[j])
        # print()
        for j in range((i + 1) * fold_size, n_samples):
            # print(j, end=' ')
            train_idx.append(indices[j])
    else:
        for j in range(i * fold_size, n_samples):
            # print(j, end=' ')
            val_idx.append(indices[j])
    # print()
    # print(train_idx)
    train_index.append(train_idx)
    val_index.append(val_idx)

## 3-2. cross validation of ridge regression

## ** step 1. calculate mse from one model with k-fold
# print("lambda:", lamb, end = ", ")
MSEs_in_current_lamb = []
train_index, val_index = kFold(X_train, num_fold)
itr = 0
for i in range(num_fold):
    # print(itr, "th fitting and validation...")

    ridge1 = ridgeRegression(lamb=lamb)

    X_train_current_fold = X_train[train_index[i]]
    y_train_current_fold = y_train[train_index[i]]

    X_validation_current_fold = X_train[val_index[i]]
    y_validation_current_fold = y_train[val_index[i]]

    ridge1.fit(X_train_current_fold, y_train_current_fold)
    pred1 = ridge1.predict(X_validation_current_fold)
    MSEs_in_current_lamb.append(mean_squared_error(y_validation_current_fold, pred1))

    # print("MSE is...", MSEs_in_current_lamb[itr])
    itr += 1
 
## ** step 2. append avg of mse of each fold to MSE_set list
# print(MSEs_in_current_lamb)
MSE_fold = sum(MSEs_in_current_lamb) / num_fold
MSE_set.append(MSE_fold)
# print("average MSE:", MSE_fold)

## ** step 3. find the best mse and lambda
# min_idx = 0
lambda_nums = len(lambdas)
for i in range(lambda_nums):
    if MSE_set[i] < MSE_set[min_idx]:
        min_idx = i
best_MSE = MSE_set[min_idx]
best_lambda = lambdas[min_idx]
# print("best_MSE:", best_MSE)
# print("best_lambda:", best_lambda)

## ** step 4. re-train ridge with best lambda
ridge = ridgeRegression(lamb=best_lambda)
# print(X_train.shape)
ridge.fit(X_train, y_train)

## ** step 5. calc mse with test data
pred_final = ridge.predict(X_test)
test_MSE = mean_squared_error(y_test, pred_final)
# print("test_MSE:", test_MSE)


## 3-3. cross validation of lasso regression

## ** step 1. calculate mse from one model with k-fold
# print("lambda:", lamb, end = ", ")
MSEs_in_current_lamb = []
train_index, val_index = kFold(X_train, num_fold)
itr = 0
for i in range(num_fold):
    # print(itr, "th fitting and validation...")

    lasso1 = Lasso(max_iter=10000)
    lasso1.set_params(alpha=lamb)

    X_train_current_fold = X_train[train_index[i]]
    y_train_current_fold = y_train[train_index[i]]

    X_validation_current_fold = X_train[val_index[i]]
    y_validation_current_fold = y_train[val_index[i]]

    lasso1.fit(X_train_current_fold, y_train_current_fold)
    pred1 = lasso1.predict(X_validation_current_fold)
    MSEs_in_current_lamb.append(mean_squared_error(y_validation_current_fold, pred1))

    # print("MSE is...", MSEs_in_current_lamb[itr])
    itr += 1

## ** step 2. append avg of mse of each fold to MSE_set list
# print(MSEs_in_current_lamb)
MSE_fold = sum(MSEs_in_current_lamb) / num_fold
MSE_set.append(MSE_fold)
# print("average MSE:", MSE_fold)

## ** step 3. find the best mse and lambda
min_idx = 0
lambda_nums = len(lambdas)
for i in range(lambda_nums):
    if MSE_set[i] < MSE_set[min_idx]:
        min_idx = i
best_MSE = MSE_set[min_idx]
best_lambda = lambdas[min_idx]
# print("best_MSE:", best_MSE)
# print("best_lambda:", best_lambda)

## ** step 4. re-train ridge with best lambda
lasso = Lasso(max_iter=10000)
lasso.set_params(alpha=best_lambda)
lasso.fit(X_train, y_train)

## ** step 5. calc mse with test data
pred_final = lasso.predict(X_test)
test_MSE = mean_squared_error(y_test, pred_final)


### 4. Confusion matrix, ROC curve

## 4-1. get confusion matrix
size = len(pred)
for i in range(size):
    if pred[i] == 1:
        if target[i] == 1:  # TP
            confu_mat[0][0] += 1
        elif target[i] == 0:  # FP
            confu_mat[0][1] += 1
    elif pred[i] == 0:
        if target[i] == 1:  # FN
            confu_mat[1][0] += 1
        elif target[i] == 0:  # TN
            confu_mat[1][1] += 1

## 4-2. get precision
tp = confu_mat[0][0]
fp = confu_mat[0][1]
precision = tp / (tp + fp)

## 4-3. recall
tp = confu_mat[0][0]
fn = confu_mat[1][0]
recall = tp / (tp + fn)

## 4-4. f1-score
prec = precision(confu_mat)
recl = recall(confu_mat)
f1_score = 2 * prec * recl / (prec + recl)

## 4-5. ROC (AUROC)
new_y_probs = y_probs[desc_score_indices]
size = len(y_true)
tpr = [0.]
fpr = [0.]
for i in range(size):
    # making prediction list by using y_probs and threshold value
    pred1 = np.zeros(size)
    threshold = new_y_probs[i]
    for j in range(size):
        if new_y_probs[j] >= threshold:
            pred1[j] = 1
        else:
            pred1[j] = 0

    # get confusion matrix
    confu_mat = Confusion_Matrix(y_true, pred1)
    tp = confu_mat[0][0]
    fp = confu_mat[0][1]
    fn = confu_mat[1][0]
    tn = confu_mat[1][1]

    tpr_in_this_threshold = tp / (tp + fn)
    fpr_in_this_threshold = fp / (fp + tn)

    tpr.append(tpr_in_this_threshold)
    fpr.append(fpr_in_this_threshold)

# print(tpr)
# print(fpr)
# where to use numpy.cumsum() ???????
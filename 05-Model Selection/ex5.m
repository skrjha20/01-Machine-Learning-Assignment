load ('ex5data1.mat');
m = size(X, 1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

function [J, grad] = linearRegCostFunction(X,y, theta, lambda)
  m=length(y);
  J = 0;
  grad = zeros(size(theta));
  theta_zeroed_first = [0; theta(2:length(theta));];
  J = (1/(2*m))*sumsq((X * theta) - y) + (lambda/(2*m))*sumsq(theta_zeroed_first);
  for (j = 1:length(grad))
    grad(j) = (1/m)*sum(((X * theta) - y).* X(:,j)) + ((lambda/m)*theta_zeroed_first(j));
  endfor
endfunction

function [theta] = trainLinearReg(X, y, lambda)
  initial_theta = zeros(size(X, 2), 1); 
  costFunction = @(t) linearRegCostFunction(X, y, t, lambda);
  options = optimset('MaxIter', 200, 'GradObj', 'on');
  theta = fmincg(costFunction, initial_theta, options);
endfunction

function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
  m = size(X, 1);
  mval = size(Xval, 1);
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);
  for i = 1:m
  	theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
	  error_train(i) = (1/(2*i))*sumsq((X(1:i,:)*theta) - y(1:i));
	  error_val(i) = (1/(2*mval))*sumsq((Xval * theta) - yval);
  endfor
endfunction

function [X_poly] = polyFeatures(X, p)
  X_poly = zeros(numel(X), p);
  for i = 1:p
	  X_poly(:, i) = X .^ i;
  endfor
endfunction

function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = zeros(size(X));
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  for i = 1:size(X,2)
    mu(i) = mean(X(:,i));
    sigma(i) = std(X(:,i));
    for j = 1:size(X,1) 
	    X_norm(j,i) = (X(:,i)(j) - mu(i))/sigma(i);
    endfor
  endfor
endfunction

function [lambda_vec, error_train_vec, error_val_vec] = validationCurve(X_poly, y, X_poly_val, yval)
  m = size(X_poly, 1);
  mpoly = size(X_poly_val, 1);
  lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
  error_train_vec = zeros(size(lambda_vec), (2));
  error_val_vec   = zeros(size(lambda_vec), (2));  
  for i = 1:length(lambda_vec)
  	theta = trainLinearReg(X_poly, y, lambda_vec(i));
	  error_train_vec(i) = (1/(2*m))*sumsq((X_poly * theta) - y);
	  error_val_vec(i) = (1/(2*mpoly))*sumsq((X_poly_val * theta) - yval);
  endfor    
endfunction

X = [ones(m, 1) X];
theta = [1 ; 1];
lambda = 1;
[J, grad] = linearRegCostFunction(X, y, theta, lambda);
[theta] = trainLinearReg(X, y, lambda);

hold on;
plot(X(:,2), X*theta, '-', 'LineWidth', 2)
hold off;

mval = size(Xval, 1);
Xval = [ones(mval, 1) Xval];
lambda = 0;
[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

p = 8;
X_poly = polyFeatures(X(:,2), p);
[X_poly, mu, sigma] = featureNormalize(X_poly);
X_poly = [ones(m, 1) X_poly];
lambda = 1;
[theta_poly] = trainLinearReg(X_poly, y, lambda);

X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];       

X_poly_val = polyFeatures(Xval(:,2), p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           

figure(1);
plot(X(:,2), y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X(:,2)), max(X(:,2)), mu, sigma, theta_poly, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

[lambda_vec, error_train_vec, error_val_vec] = validationCurve(X_poly, y, X_poly_val, yval);
plot(lambda_vec, error_train_vec, lambda_vec, error_val_vec);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train_vec(i), error_val_vec(i));
end

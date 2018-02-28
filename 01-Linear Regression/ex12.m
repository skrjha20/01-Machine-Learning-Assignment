data = load('ex1data2.txt');
X1 = data(:,1); 
X2 = data(:, 2); 
y = data(:, 3);

X1 = (X1-mean(X1))/std(X1);
X2 = (X2-mean(X2))/std(X2);
y  = (y-mean(y))/std(y);

m = length(y); 
X = [ones(m, 1), X1, X2]; 

function J = computeCost(X, y, theta)  
  m = length(y);
  J = 0;
  J = (1./(2*m))*sum(((X * theta) - y) .^ 2);
end

function theta = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  J1 = 0;
  for iter = 1:num_iters
    J2 = computeCost(X, y, theta);
    if abs(J2-J1) == 0
      break
    end
    grad = (alpha/m)*sum(X.*repmat((X*theta - y), 1, size(X,2)));
    theta = (theta' - grad)';
    J2 = J1;
  endfor
end

theta = zeros(3,1);
iterations = 1000;
alpha = 0.0001;
J2=0;
iterations = 1500;
alpha = 0.01;
computeCost(X, y, theta);
theta = gradientDescent(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent: \n');
fprintf('%f \n', theta);

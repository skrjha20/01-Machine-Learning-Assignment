data = load('ex1data1.txt');
data = load('ex1data1.txt');
X = data(:, 1); 
y = data(:, 2);
m = length(y); 

X = [ones(m, 1), data(:,1)]; 
theta = zeros(2,1);
iterations = 1000;
alpha = 0.0001;
J2=0;

function plotData(x, y)
  figure;
  plot(x, y, 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  ylabel('Profix in $10,000s');
  xlabel('Population of City in 10,000s');
end

function J = computeCost(X, y, theta)  
  m = length(y);
  J = 0;
  J = (1./(2*m))*sum(((X * theta) - y) .^ 2);
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
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

fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);
plotData(X, y);
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Running Gradient Descent ...\n')
X = [ones(m, 1), data(:,1)]; 
theta = zeros(2, 1);

iterations = 1500;
alpha = 0.01;
computeCost(X, y, theta);
theta = gradientDescent(X, y, theta, alpha, iterations);

fprintf('Theta found by gradient descent: \n');
fprintf('%f\n', theta);

hold on; 
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off 



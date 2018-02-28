data = load('ex2data2.txt');
X = data(:, 1:2);
y  = data(:, 3);
m = length(y);
pos = find(y==1);
neg = find(y==0);

function plotData(X, y)
  figure; hold on;
  pos = find(y==1); 
  neg = find(y==0);
  plot(X(pos,1),X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  hold on;
  plot(X(neg,1),X(neg,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  xlabel('Microchip Test1')
  ylabel('Microchip Test2')
  hold off;
endfunction

function out = mapFeature(X1, X2)
  degree = 6;
  out = ones(size(X1(:,1)));
  for i = 1:degree
    for j = 0:i
      out(:, end+1) = (X1.^(i-j)).*(X2.^j);      
    endfor
  endfor
endfunction

function g = sigmoid(z)
  g = 1./(1 + exp(-z));
endfunction

function [JVal,gradient] = costFunctionReg(theta,X,y,lambda)
  m=length(y);
  JVal = (-1./m)*sum(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta))) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta)));
  gradient = (1./m)*sum(X.*repmat((sigmoid(X*theta) - y), 1, size(X,2)));
  gradient(:,2:length(gradient)) = gradient(:,2:length(gradient)) + (lambda/m)*theta(2:length(theta))';
endfunction

function plotDecisionBoundary(theta, X, y)
  hold on
  u = linspace(-1, 1.5, 50);
  v = linspace(-1, 1.5, 50);
  z = zeros(length(u), length(v));
  for i = 1:length(u)
    for j = 1:length(v)
      z(i,j) = mapFeature(u(i), v(j))*theta;
    endfor
  endfor
  z = z'; 
  contour(u, v, z, [0, 0], 'LineWidth', 2);
  legend('y = 1', 'y = 0', 'Decision boundary')
  hold off;
endfunction

plotData(X,y)
X = mapFeature(X(:,1), X(:,2));
theta = ones(size(X, 2), 1);
lambda = 0;
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, JVal] = fminunc(@(t)(costFunctionReg(t,X,y,lambda)), theta, options);

fprintf('Theta found by gradient descent: \n');
fprintf('%f\n', theta);

fprintf('\nCost: ');
fprintf('%f \n', JVal);

plotDecisionBoundary(theta, X, y);

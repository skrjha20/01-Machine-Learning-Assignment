data = load('ex2data1.txt');
X = data(:, 1:2);
y  = data(:, 3);
m = length(y);

function plotData(X, y)
  figure; hold on;
  pos = find(y==1); 
  neg = find(y==0);
  plot(X(pos,1),X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  hold on;
  plot(X(neg,1),X(neg,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  xlabel('Exam 1 score');  
  ylabel('Exam 2 score');
  hold off;
endfunction
  
function g = sigmoid(z)
	g = 1./(1 + exp(-z));
endfunction

function p = predict(theta,X)
	p=sigmoid(X*theta)>=0.5;
endfunction

function [JVal,grad] = costFunction(theta,X,y)
	m=length(y);
	alpha = 0.0001;
	JVal = (-1/m)*sum(y'*log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta)));
	grad = (alpha/m)*sum(X.*repmat((sigmoid(X*theta) - y), 1, size(X,2)));
endfunction

function plotDecisionBoundary(theta, X, y)
  hold on
  plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
  plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
  plot(plot_x, plot_y);
  legend('Admitted', 'Not admitted', 'Decision Boundary');
  axis([30, 100, 30, 100]);
  hold off
endfunction

plotData(X,y);
X = [ones(m, 1), X];
theta = zeros(3,1); 
options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta, JVal] = fminunc(@(t)(costFunction(t,X,y)), theta, options);

fprintf('Theta found by gradient descent: \n');
fprintf('%f\n', theta);

fprintf('\nCost: ');
fprintf('%f \n', JVal);

plotDecisionBoundary(theta, X, y)

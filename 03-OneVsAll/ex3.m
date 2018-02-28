load('ex3data1.mat');
m = size(X, 1);
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

function [h, display_array] = displayData(X, example_width)
  if ~exist('example_width', 'var') || isempty(example_width) 
	  example_width = round(sqrt(size(X, 2)));
  endif
  colormap(gray);
  [m n] = size(X);
  example_height = (n / example_width);
  display_rows = floor(sqrt(m));
  display_cols = ceil(m / display_rows);
  pad = 1;
  display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));  
  curr_ex = 1;
  for j = 1:display_rows
	  for i = 1:display_cols
		  if curr_ex > m, 
			  break; 
		  endif
      max_val = max(abs(X(curr_ex, :)));
		  display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		  curr_ex = curr_ex + 1;
	  endfor
	  if curr_ex > m, 
		  break; 
	  endif
  endfor
  h = imagesc(display_array, [-1 1]);
  axis image off
  drawnow;
endfunction

function g = sigmoid(z)
  g = 1./(1 + exp(-z));
endfunction

function [J, grad] = lrCostFunction1(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  theta_reg = [0;theta(2:end, :);];
  J = (-1./m)*(y'* log(sigmoid(X*theta)) + (1-y)'*log(1-sigmoid(X*theta)))+(lambda/(2*m))*theta_reg'*theta_reg;
  grad = (1./m)*(X'*(sigmoid(X*theta)-y)+(lambda/m)*theta_reg);
endfunction

function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  J = -(1./m)*sum(y'*log(sigmoid(X*theta)) + (1-y')*log(1 - sigmoid(X*theta)));
  grad = (1./m)*(X'*(sigmoid(X*theta) - y)); 
  theta_zeroed_first = [0; theta(2:length(theta));];
  J = J + lambda / (2 * m) * sum( theta_zeroed_first .^ 2 );
  grad = grad .+ (lambda / m) * theta_zeroed_first;
  grad = grad(:);
endfunction

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_labels, n + 1);
  X = [ones(m, 1) X];
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for c = 1:num_labels
    all_theta(c,:) = fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
  endfor
endfunction

function g = sigmoid(z)
  g = 1./(1 + exp(-z));
endfunction

function [J, grad] = lrCostFunction1(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  theta_reg = [0;theta(2:end, :);];
  J = (-1./m)*sum(y'*log(sigmoid(X*theta)) + (1-y')*log(1-sigmoid(X*theta)))+(lambda/(2*m))*theta_reg'*theta_reg;
  grad = (1./m)*sum(X'*(sigmoid(X*theta)-y))+(lambda/m)*theta_reg;
endfunction

function [J, grad] = lrCostFunction(theta, X, y, lambda)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  J = -(1./m)*sum(y'*log(sigmoid(X*theta)) + (1-y')*log(1 - sigmoid(X*theta)));
  grad = (1./m)*(X'*(sigmoid(X*theta) - y)); 
  theta_zeroed_first = [0; theta(2:length(theta));];
  J = J + lambda / (2 * m) * sum( theta_zeroed_first .^ 2 );
  grad = grad .+ (lambda / m) * theta_zeroed_first;
  grad = grad(:);
endfunction
 
function p = predictOneVsAll(all_theta, X)
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  p = zeros(size(X, 1), 1);
  X = [ones(m, 1) X];
  [max_value, p] = max(sigmoid(X*all_theta'), [], 2);
endfunction

function p = predict1(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  a1 = [ones(m, 1) X];
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m, 1) a2];
  z3 = a2 * Theta2';
  [max_value, p] = max(sigmoid(z3), [], 2);
endfunction

function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  z3 = [ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2';
  [max_value, p] = max(sigmoid(z3), [], 2);
endfunction

displayData(sel);
input_layer_size  = 400; 
num_labels = 10; 
lambda = 1;

[all_theta] = oneVsAll(X, y, num_labels, lambda);
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
load('ex3weights.mat');
pred_nn = predict(Theta1, Theta2, X);

rp = randperm(m);
for i = 1:5
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

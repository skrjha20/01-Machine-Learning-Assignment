load('ex4data1.mat');
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

function g = sigmoidGradient(z)
  g = zeros(size(z));
  g = sigmoid(z).*(1 - sigmoid(z));
endfunction

function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  m = length(y);
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  K = num_labels;
  X = [ones(m,1) X];
  for i = 1:m
    a1_i = X(i,:);
    z2_i = a1_i * Theta1';
    a2_i = sigmoid(z2_i);
    a2_i = [1 a2_i];
    z3_i = a2_i * Theta2';
    h_of_Xi = sigmoid(z3_i);
    y_i = zeros(1,K);
    y_i(y(i)) = 1;
    J = J + (-1/m)*sum(y_i.*log(h_of_Xi) + (1-y_i).*log(1-h_of_Xi));
  endfor
  J = J + (lambda/(2*m))*(sum(sumsq(Theta1(:,2:input_layer_size+1))) + sum(sumsq(Theta2(:,2:hidden_layer_size+1)))) ;         
  delta_accum_1 = zeros(size(Theta1));
  delta_accum_2 = zeros(size(Theta2));
  for i = 1:m
	  a1_i = X(i,:);  
	  z2_i = a1_i * Theta1';
    a2_i = sigmoid(z2_i);
    a2_i = [1 a2_i];
    z3_i = a2_i * Theta2';
    a3_i = sigmoid(z3_i);   
    y_i = zeros(1,K);
    y_i(y(i)) = 1;
    
    delta_3 = a3_i - y_i;
    delta_2 = delta_3*Theta2 .* sigmoidGradient([1 z2_i]);    
    delta_accum_1 = delta_accum_1 + delta_2(2:end)' * a1_i;
    delta_accum_2 = delta_accum_2 + delta_3' * a2_i;
  endfor
  size(delta_3)
  size(delta_2)
  Theta1_grad = delta_accum_1 / m;
  Theta2_grad = delta_accum_2 / m;
  Theta1_grad(:, 2:input_layer_size+1) = Theta1_grad(:, 2:input_layer_size+1) + lambda / m * Theta1(:, 2:input_layer_size+1);
  Theta2_grad(:, 2:hidden_layer_size+1) = Theta2_grad(:, 2:hidden_layer_size+1) + lambda / m * Theta2(:, 2:hidden_layer_size+1);
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
endfunction   

function numgrad = computeNumericalGradient(J, theta)
  numgrad = zeros(size(theta));
  perturb = zeros(size(theta));
  e = 1e-4;
  for p = 1:numel(theta)
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    numgrad(p) = (loss2 - loss1)/(2*e);
    perturb(p) = 0;
  endfor
endfunction
  
function checkNNGradients(lambda)
  if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
  end
  input_layer_size = 3;
  hidden_layer_size = 5;
  num_labels = 3;
  m = 5;
  Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
  Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
  X  = debugInitializeWeights(m, input_layer_size - 1);
  y  = 1 + mod(1:m, num_labels)';
  nn_params = [Theta1(:) ; Theta2(:)];
  J = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
  [cost, grad] = J(nn_params);
  numgrad = computeNumericalGradient(J, nn_params);
  disp([numgrad grad]);
  fprintf(['The above two columns you get should be very similar.\n' ...
           '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']); 
  diff = norm(numgrad-grad)/norm(numgrad+grad);
  fprintf(['If your backpropagation implementation is correct, then \n' ...
           'the relative difference will be small (less than 1e-9). \n' ...
           '\nRelative Difference: %g\n'], diff);
endfunction

function p = predict(Theta1, Theta2, X)
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  p = zeros(size(X, 1), 1);
  z3 = [ones(m, 1) sigmoid([ones(m, 1) X] * Theta1')] * Theta2';
  [max_value, p] = max(sigmoid(z3), [], 2);
endfunction

function W = randInitializeWeights(L_in, L_out)
  W = zeros(L_out, 1 + L_in);
  epsilon_init = 0.12;
  W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end

function W = debugInitializeWeights(fan_out, fan_in)
  W = zeros(fan_out, 1 + fan_in);
  W = reshape(sin(1:numel(W)), size(W)) / 10;
end

displayData(sel);
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];
input_layer_size  = 400;
hidden_layer_size = 25;
num_labels = 10; 

lambda = 0;                 
[J1, grad1] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

lambda = 1;                 
[J2, grad2] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
lambda=3;
checkNNGradients(lambda);
pred_nn = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_nn == y)) * 100);

rp = randperm(m);
for i = 1:5
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));
    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end
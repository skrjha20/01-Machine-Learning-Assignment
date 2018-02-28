fprintf('Visualizing example dataset for PCA.\n\n');
load ('ex7data1.mat');

function [X_norm, mu, sigma] = featureNormalize(X)
  mu = mean(X);
  X_norm = bsxfun(@minus, X, mu);
  sigma = std(X_norm);
  X_norm = bsxfun(@rdivide, X_norm, sigma);
endfunction

function [U, S] = pca(X)
  [m, n] = size(X);
  U = zeros(n);
  S = zeros(n);
  Sigma = (X' * X) / m; 
  [U, S, V] = svd(Sigma);
endfunction

function drawLine(p1, p2, varargin)
  plot([p1(1) p2(1)], [p1(2) p2(2)], varargin{:});
endfunction

function Z = projectData(X, U, K)
  Z = zeros(size(X, 1), K);
  for i = 1:size(Z,1)
	  x = X(i, :)';
	  Z(i,:) = x' * U(:, 1:K);
  endfor
endfunction

function X_rec = recoverData(Z, U, K)
  X_rec = zeros(size(Z, 1), size(U, 1));
  for i = 1:size(Z,1)
	  v = Z(i, :)';
	  X_rec(i,:) = v' * U(:, 1:K)';
  endfor	
endfunction

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

plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning PCA on example dataset.\n\n');
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);

hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nDimension reduction on example dataset.\n\n');
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square

K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));

X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));

hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nLoading face dataset.\n\n');
load ('ex7faces.mat')
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf(['\nRunning PCA on face dataset. This mght take a minute or two ...\n\n']);
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
displayData(U(:, 1:36)');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nDimension reduction for face dataset.\n\n');
K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));
fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');
K = 100;
X_rec  = recoverData(Z, U, K);

subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;
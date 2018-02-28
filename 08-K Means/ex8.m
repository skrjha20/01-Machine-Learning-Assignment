fprintf('Finding closest centroids.\n\n');
load('ex7data2.mat');

function idx = findClosestCentroids(X, centroids) 
  K = size(centroids, 1);
  idx = zeros(size(X,1), 1);
  for i = 1:length(idx)
	  mindist = realmax();
	  for k = 1:K
		  d = norm(X(i,:) - centroids(k,:));
		  if (d < mindist)
			  mindist = d;
			  idx(i) = k;
		  endif
	  endfor
  endfor  
endfunction

function centroids = computeCentroids(X, idx, K)
  [m n] = size(X);
  centroids = zeros(K, n);
  for k = 1:K
	  idx_k = find(idx==k);
	  centroids(k,:) = sum(X(idx_k,:),1) / length(idx_k);
  endfor
endfunction

function [centroids, idx] = runkMeans(X, initial_centroids, max_iters, plot_progress)
  if ~exist('plot_progress', 'var') || isempty(plot_progress)
    plot_progress = false;
  endif  
  if plot_progress
    figure;
    hold on;
  endif
  
  [m n] = size(X);
  K = size(initial_centroids, 1);
  centroids = initial_centroids;
  previous_centroids = centroids;
  idx = zeros(m, 1);

  for i=1:max_iters
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    if exist('OCTAVE_VERSION')
       fflush(stdout);
    endif
    idx = findClosestCentroids(X, centroids);
    if plot_progress
       plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
       previous_centroids = centroids;
       fprintf('Press enter to continue.\n');
       pause;
    endif
    centroids = computeCentroids(X, idx, K);
  endfor
  if plot_progress
    hold off;
  endif
endfunction

function centroids = kMeansInitCentroids(X, K)
  centroids = zeros(K, size(X, 2));
  randidx = randperm(size(X, 1));
  centroids = X(randidx(1:K), :);
endfunction

K = 3;
initial_centroids = [3 3; 6 2; 8 5];
idx = findClosestCentroids(X, initial_centroids);
fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nComputing centroids means.\n\n');
centroids = computeCentroids(X, idx, K);
fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning K-Means clustering on example dataset.\n\n');
load('ex7data2.mat');
K = 3;
max_iters = 10;
initial_centroids = [3 3; 6 2; 8 5];
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, false);
fprintf('\nK-Means Done.\n\n');

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');
A = double(imread('bird_small.png'));
A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, false);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nApplying K-Means to compress an image.\n\n');
idx = findClosestCentroids(X, centroids);
X_recovered = centroids(idx,:);
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

subplot(1, 2, 1);
imagesc(A); 
title('Original');

subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));

fprintf('Program paused. Press enter to continue.\n');
pause;

sel = floor(rand(1000, 1) * size(X, 1)) + 1;
palette = hsv(K);
colors = palette(idx(sel), :);
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
fprintf('Program paused. Press enter to continue.\n');
pause;

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

function Z = projectData(X, U, K)
  Z = zeros(size(X, 1), K);
  for i = 1:size(Z,1)
	  x = X(i, :)';
	  Z(i,:) = x' * U(:, 1:K);
  endfor
endfunction

figure;
Z = projectData(X_norm, U, 2);
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
fprintf('Program paused. Press enter to continue.\n');
pause;
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
#Z = projectData(X_norm, U, 2);


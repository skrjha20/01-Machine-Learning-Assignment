fprintf('Visualizing example dataset for outlier detection.\n\n');
load('ex8data1.mat');

plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause

function [mu sigma2] = estimateGaussian(X)
  [m, n] = size(X);
  mu = zeros(n, 1);
  sigma2 = zeros(n, 1);
  mu = mean(X);
  sigma2 = var(X).*(m-1)./m;
endfunction

function p = multivariateGaussian(X, mu, Sigma2)
  k = length(mu);
  if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
  endif
  X = bsxfun(@minus, X, mu(:)');
  p = (2*pi)^(-k/2)*det(Sigma2)^(-0.5)*exp(-0.5*sum(bsxfun(@times, X*pinv(Sigma2),X), 2));
endfunction

function visualizeFit(X, mu, sigma2)
  [X1,X2] = meshgrid(0:.5:35); 
  Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
  Z = reshape(Z,size(X1));
  plot(X(:, 1), X(:, 2),'bx');
  hold on;
  if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z, 10.^(-20:3:0)');
  endif
  hold off;
endfunction

function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;
  stepsize = (max(pval) - min(pval)) / 1000;
  for epsilon = min(pval):stepsize:max(pval)
    pred = (pval < epsilon);
	  tp = sum((pred == 1) & (yval == 1));
	  fp = sum((pred == 1) & (yval == 0));
	  fn = sum((pred == 0) & (yval == 1));
	  precision = tp/(tp + fp);
	  recall = tp/(tp + fn);
  	F1 = (2*precision*recall)/(precision + recall);
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    endif
  endfor
endfunction
  
fprintf('Visualizing Gaussian fit.\n\n');
[mu sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;

pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

outliers = find(p < epsilon);
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

load('ex8data2.mat');
[mu sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));
pause



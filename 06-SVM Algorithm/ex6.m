load('ex6data1.mat');

function plotData(X, y)
  pos = find(y == 1); neg = find(y == 0);
  plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
  hold on;
  plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
  hold off;
endfunction

function sim = linearKernel(x1, x2)
  x1 = x1(:); x2 = x2(:);
  sim = x1' * x2;
endfunction

function visualizeBoundaryLinear(X, y, model)
  w = model.w;
  b = model.b;
  xp = linspace(min(X(:,1)), max(X(:,1)), 100);
  yp = - (w(1)*xp + b)/w(2);
  plotData(X, y);
  hold on;
  plot(xp, yp, '-b'); 
  hold off
endfunction

plotData(X, y);
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

function sim = gaussianKernel(x1, x2, sigma)
  x1 = x1(:); x2 = x2(:);
  sim = 0;
  sim = exp(-sum((x1-x2).^2)/(2*(sigma ^ 2)));
endfunction

fprintf('\nEvaluating the Gaussian Kernel ...\n')
x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

function visualizeBoundary(X, y, model, varargin)
  plotData(X, y)
  x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
  x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
  [X1, X2] = meshgrid(x1plot, x2plot);
  vals = zeros(size(X1));
  for i = 1:size(X1, 2)
    this_X = [X1(:, i), X2(:, i)];
    vals(:, i) = svmPredict(model, this_X);
  endfor
  hold on
  contour(X1, X2, vals, [0 0], 1 , 'linecolor', 'blue');
  hold off;
endfunction
         
load('ex6data2.mat');
plotData(X, y);
C = 1;
sigma = 0.1;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

function [C, sigma] = dataset3Params(X, y, Xval, yval)  
  possibles = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
  error = zeros(size(possibles, 2),size(possibles, 2));
  for i=1:size(possibles, 2)
    for j=1:size(possibles, 2)
      C_pos = possibles(i);
      S_pos = possibles(j);
      model = svmTrain(X, y, C_pos, @(x1, x2) gaussianKernel(x1, x2, S_pos));
      y_pred = svmPredict(model, Xval);
      error(i,j) = mean(double(y_pred ~= yval));
    endfor
  endfor
  [r,c] = find(error == min(min(error)));
  C = possibles(r);
  sigma = possibles(c);
endfunction

load('ex6data3.mat');
plotData(X, y);
[C, sigma] = dataset3Params(X, y, Xval, yval);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
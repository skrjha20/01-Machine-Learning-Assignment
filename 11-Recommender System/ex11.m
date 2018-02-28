fprintf('Loading movie ratings dataset.\n\n');
load ('ex8_movies.mat');

fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));
  D = (R .* (X * Theta' - Y));
  J = 1 / 2 * sum(sum( D .^ 2 ));
  J = J + (lambda / 2) * (sum(sumsq(Theta)) + sum(sumsq(X))); 
  for i = 1:num_movies
	  idx = find(R(i,:) == 1);
	  theta_tmp = Theta(idx,:);
	  y_tmp = Y(i,idx);
	  X_grad(i,:) = ((X(i,:) * theta_tmp') - y_tmp) * theta_tmp + lambda * X(i,:);
  endfor
  for j = 1:num_users
    idx = find(R(:,j) == 1);
	  x_tmp = X(idx,:);
	  y_tmp = Y(idx,j);
	  Theta_grad(j,:) = ((Theta(j,:) * x_tmp') - y_tmp') * x_tmp + lambda * Theta(j,:);
  endfor
  grad = [X_grad(:); Theta_grad(:)];
endfunction

function numgrad = computeNumericalGradient(J, theta)
  numgrad = zeros(3,38);
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

function checkCostFunction(lambda)
  if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
  endif
  X_t = rand(4, 3);
  Theta_t = rand(5, 3);
  Y = X_t * Theta_t';
  Y(rand(size(Y)) > 0.5) = 0;
  R = zeros(size(Y));
  R(Y ~= 0) = 1;
  X = randn(size(X_t));
  Theta = randn(size(Theta_t));
  num_users = size(Y, 2);
  num_movies = size(Y, 1);
  num_features = size(Theta_t, 2);
  numgrad = computeNumericalGradient(@(t) cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda), [X(:); Theta(:)])
  size(numgrad)
  [cost, grad] = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies, num_features, lambda);
  size(grad)
  disp([numgrad grad]);
  fprintf(['The above two columns you get should be very similar. Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
  diff = norm(numgrad-grad)/norm(numgrad+grad);
endfunction

function [Ynorm, Ymean] = normalizeRatings(Y, R)
  [m, n] = size(Y);
  Ymean = zeros(m, 1);
  Ynorm = zeros(size(Y));
  for i = 1:m
    idx = find(R(i, :) == 1);
    Ymean(i) = mean(Y(i, idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
  endfor
endfunction

function movieList = loadMovieList()
  fid = fopen('movie_ids.txt');
  n = 1682;  % Total number of movies 
  movieList = cell(n, 1);
  for i = 1:n
    line = fgets(fid);
    [idx, movieName] = strtok(line, ' ');
    movieList{i} = strtrim(movieName);
  endfor
  fclose(fid);
endfunction

load ('ex8_movieParams.mat');
%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 0);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Gradients (without regularization) ... \n');
#checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);             
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
fprintf('\nChecking Gradients (with regularization) ... \n');
#checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

movieList = loadMovieList();
my_ratings = zeros(1682, 1);
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    endif
endfor

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nTraining collaborative filtering...\n');
load('ex8_movies.mat');
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];
[Ynorm, Ymean] = normalizeRatings(Y, R);
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];
options = optimset('GradObj', 'on', 'MaxIter', 100);
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda)), initial_parameters, options);
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

fprintf('Recommender system learning completed.\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

p = X * Theta';
my_predictions = p(:,1) + Ymean;
movieList = loadMovieList();
[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
  j = ix(i);
  fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
endfor

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
  if my_ratings(i) > 0 
    fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
  endif
endfor


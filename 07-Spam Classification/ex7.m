fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

function vocabList = getVocabList()
  fid = fopen('vocab.txt');
  n = 1899;
  vocabList = cell(n, 1);
  for i = 1:n
    fscanf(fid, '%d', 1);
    vocabList{i} = fscanf(fid, '%s', 1);
  endfor
  fclose(fid);
endfunction

function word_indices = processEmail(email_contents)
  vocabList = getVocabList();
  word_indices = [];
  email_contents = lower(email_contents);
  email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
  email_contents = regexprep(email_contents, '[0-9]+', 'number');
  email_contents = regexprep(email_contents, '(http|https)://[^\s]*', 'httpaddr');
  email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');
  email_contents = regexprep(email_contents, '[$]+', 'dollar');
  fprintf('\n==== Processed Email ====\n\n');
  l = 0;
  while ~isempty(email_contents)
    [str, email_contents] = strtok(email_contents, [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
    str = regexprep(str, '[^a-zA-Z0-9]', '');
    try str = porterStemmer(strtrim(str)); 
    catch str = ''; continue;
    end;
    
    if length(str) < 1
      continue;
    end          
    for i=1:length(vocabList)
	    match(i)=strcmp(vocabList{i}, str);
  		if (match(i)==1)
	  		word_indices = [word_indices; i];
		  else
			  word_indices = [word_indices;[]];
		  endif
	  endfor
    if (l + length(str) + 1) > 78
      fprintf('\n');
      l = 0;
    endif
    fprintf('%s ', str);
    l = l + length(str) + 1;
  endwhile
  fprintf('\n\n=========================\n');
endfunction

function x = emailFeatures(word_indices)
  n = 1899;
  x = zeros(n, 1);
  for i=1:length(word_indices)
	  x(word_indices(i)) = 1;
  endfor
endfunction

function sim = linearKernel(x1, x2)
  x1 = x1(:); x2 = x2(:);
  sim = x1' * x2;
endfunction

function sim = gaussianKernel(x1, x2, sigma)
  x1 = x1(:); x2 = x2(:);
  sim = 0;
  sim = exp(-sum((x1-x2).^2)/(2*(sigma ^ 2)));
endfunction

file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');
features = emailFeatures(word_indices);
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;

[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

filename = 'spamSample1.txt';

file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

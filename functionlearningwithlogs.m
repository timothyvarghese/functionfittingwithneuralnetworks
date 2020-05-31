%% Goal of code: since shallow networks work badly for polynomials greater than x^2
%% this code does log learning to convert the high growing functions into linear or sublinear
%% functions and recovers them by using exp(net(i).  Since these are fast growing functions
%% we use relative mean square error and not MSE.

nnstart
clear all;
close all;

%% paremeters
testpoint = 4 % sanity check for a testpoint to compare learned and actual function
trainingSize = 100; % how many points in training set
trainingStart = 1; % start of training range
trainingEnd = 10; % end of training range
hiddenUnits = 10; 
step = (trainingEnd - trainingStart)/trainingSize; 
%step = 1; % step between points in training set
testStart = 1; % start of training range
testRange = 10; % training range
testSize = 100;

%% Initialization
x = zeros(1, trainingSize);
t = zeros(1, trainingSize);

%% Training
for i = 1: trainingSize
    x(i) = i * step;
end;
for i  = 1: 100
    t(i) = log(learnedf(i * step)); % learn log of learned function
end;

net = fitnet(hiddenUnits);
net = train(net, x, t);
y = net(x);
perf = perform(net, x, t);

%% Now start testing

mse= 0; % initialize mse
normalizedmse = 0; % initialize normalized mse

for j  = 1:testSize % 
    i = testStart + rand * testRange; %random test point in test range  
    testoutput = learnedf(i);
    n=net(i); % Recall net is trained to give log, so we have to recover using exp(n)
    normalizederror =(exp(n)-testoutput)/testoutput;
    error = exp(n) - testoutput;
    normalizedmse=normalizedmse+normalizederror * normalizederror;
    mse = mse + error * error;
end;

normmse=normalizedmse/testSize;
mse = mse/testSize;
fprintf("The normalized Mean Square Error is %5.7f\n", normmse);
fprintf("The Mean Square Error is %5.7f\n", mse);

%testpoint = 4;  % answer should be 256 for x^4 and 64 for x^3
answer = exp(net(testpoint));
fprintf("For test point 4, learned answer is %5.4f and actual answer is %5.4f \n", answer, learnedf(testpoint));

%% Define function to to be learned here, some sample functions are shown
function f = learnedf(input) % changed learned function only here
%f = input * input * input * input; % x^4
%f = input * input * input; % x^3
%f = input * input; % x^2
f = exp(input); % e^x
end
%% Goal: do sine learning on 1 Hz and 0 phase from 1 to 2 pi and transfer learning to 
%% any frequency, phase, and also to any real positive number input in presence of noise

%% Set up
nnstart % Start the neural network
targetdegreephase = 0; % want to model a sine of this phase
targetfrequency = 1;  % want to model a sine of this frequence
targetphase = pi * targetdegreephase/180; % target phase in radians

trainedfrequency=1; % this is the frequency we use for training, then we scale
trainingRange = 6.28; % domain of 1 to 2 pi used for training
hiddenlayers = 10; % we know 10 units is fine for this frequency
trainingsize = 1000; % Number of training points, we know 100 works well
noiseMax = 1.0
testsize = 200; % we will compare this many test points

%% train network on 1 Hz and 0 phase
step = trainingRange/trainingsize;   % training Points are spaced apart by this step size
x = zeros(1, trainingsize); % Initialize the X-points 
t = zeros(1, trainingsize); % Initialize the training output
for i = 1: trainingsize
    x(i) = i * step; % Increase x in increments of step
end;
for i  = 1: trainingsize % This is where we write the training function
    t(i) = sin (i * step*trainedfrequency) + (rand - 0.5) * noiseMax; % train on base frequency and phase 0
end;

figure('Name', 'Regression Network', 'NumberTitle', 'off'); hold on;
plot(x, t, 'mx', 'linewidth', 1);

net = fitnet(hiddenlayers); % Create a neural network of 10 hidden layers
%training algorithms
net.trainFcn = 'trainbr'; % Bayesian, smoothest 
%net.trainFcn = 'trainlm'; LM algorithm
%net.trainFcn = 'trainscg'; % Scaled Conjugate Gradient

net = train(net, x, t); % Train the network on the training set (x,t)
y_pred = net(x);

plot(x, y_pred, 'g', 'linewidth', 2);
legend('training data', 'underlying function', 'network output');

%% Test, own test besides Matlab perform so we can try on data not in training set
sse=0; % Sum of Squared Errors initialized to 0 for test
for i  = 1: testsize % test phase
    input = rand * 2 * pi; % pick a random value in trained range
    testoutput = sin(input * (targetfrequency/trainedfrequency) + targetphase);
    % create test output as a random sine of target frequency and phase
    networkoutput=net(mod(input * (targetfrequency/trainedfrequency) + targetphase, 2 * pi));
    % Apply network, scale and divide by period to stay in trained domain
    error=networkoutput-testoutput; % calculate error
    sse=sse+error * error; % calculate squared error
end;
mse =sse/trainingsize;
fprintf ("Target Phase was %3d\n", targetdegreephase);
fprintf("For target frequency %3d Hz, %4d training points, and %2d layers, MSE was %2.8f%\n", targetfrequency, trainingsize, hiddenlayers, mse);
fprintf ("\n");


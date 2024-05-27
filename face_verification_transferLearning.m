clc;
close all;
clear all;

% Load pre-trained SqueezeNet
net = load('Face_verification_Network.mat');

% Define the main dataset directory
datasetDir = 'D:\Aks\AKS_SYSTEM\Computer vision based research\My_work\Databases\Proposed_FER\IITD_FER_Flat';

% Create an imageDatastore for the entire dataset
imds = imageDatastore(datasetDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Modify the network for transfer learning
numClasses = numel(categories(imds.Labels));
net=net.net;
inputSize = net.Layers(1).InputSize;
% Allow the user to select test images
[testFileName, testFilePath] = uigetfile({'*.jpg;*.png;*.bmp;*.tif', 'Image Files (*.jpg, *.png, *.bmp, *.tif)'; '*.*', 'All Files (*.*)'}, 'Select Test Images', 'MultiSelect', 'on');
if isequal(testFileName, 0)
    error('No files selected.');
end
testImagePath = fullfile(testFilePath, testFileName);
testImage = imread(testImagePath);
testImage = imresize(testImage, inputSize(1:2));
testFeatures = activations(net, testImage, 'relu_conv10', 'OutputAs', 'rows');

% Extract features for all dataset images
datasetFeatures = activations(net, imds, 'relu_conv10', 'OutputAs', 'rows');

% Compute cosine similarity between test images and dataset images

similarityScores = datasetFeatures * testFeatures' ./ (vecnorm(datasetFeatures, 2, 2) * norm(testFeatures));

% Find the top 5 most similar images
[~, idx] = sort(similarityScores, 'descend');
top5Idx = idx(1:5);

% Display the top 5 similar images
figure;
for i = 1:5
    subplot(1, 5, i);
    imshow(readimage(imds, top5Idx(i)));
    title(sprintf('Similarity: %.2f', similarityScores(top5Idx(i))));
end

disp('Top 5 similar images displayed.');
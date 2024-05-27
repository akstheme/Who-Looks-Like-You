clc;
close all;
clear all;
% Load pre-trained network
net = squeezenet; % You can use other networks like vgg16, vgg19, etc.
inputSize = net.Layers(1).InputSize;

% Define directories
datasetDir = 'D:\Aks\AKS_SYSTEM\Computer vision based research\My_work\Databases\Proposed_FER\IITD_FER_Flat';
subjectFolders = dir(fullfile(datasetDir, 'Subject*'));

% Create a datastore for the images
imds = imageDatastore(fullfile(datasetDir, '**', '*.jpg'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize and preprocess images, and extract features
augimds = augmentedImageDatastore(inputSize(1:2), imds, 'ColorPreprocessing', 'gray2rgb');
features = activations(net, augimds, 'fc1000', 'OutputAs', 'rows');

% Read the test image, resize it, and extract features
testImage = imread('path_to_test_image.jpg');
testImage = imresize(testImage, inputSize(1:2));
testImage = repmat(testImage, [1 1 3]); % Ensure it has 3 channels if grayscale
testFeature = activations(net, testImage, 'fc1000', 'OutputAs', 'rows');

% Compute cosine similarity between test image and dataset images
similarityScores = features * testFeature' ./ (vecnorm(features, 2, 2) * norm(testFeature));

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

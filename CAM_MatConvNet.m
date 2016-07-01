function [ output_args ] = CAM_MatConvNet( input_args )
%% CAM_IMPROVED CAM on Pascal VOC
%% Created by Hong-Yu Zhou at 2016.6.03
%% I made some changes to achieve a better result on pascal voc

% path to matconvnet
run ./matconvnet-1.0-beta20/matlab/vl_setupnn
addpath ./matconvnet-1.0-beta20/matlab/
addpath ./matconvnet-1.0-beta20/examples/

% set parameters
opts.dataDir = fullfile('./data/') ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.expDir = opts.dataDir;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 32 ;
opts.train.numEpochs = 20 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [12] ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir;
opts.train.learningRate=logspace(-2.5,-5,20); % fine-tuned

classNum = 20 ; % 20 classes in Pascal VOC 2007

% load model
net = load('./vgg_net/imagenet-vgg-verydeep-16.mat') ;

% load dataset
load('imdb.mat') ;

f=0.01 ;
net.layers(31:end) = [] ;

net.layers{end+1} = struct('type','conv','weights',{{f*randn(3,3,512,1024,'single'),zeros(1,1024,'single')}},...
    'stride',1,'pad', [1 1 1 1],'filtersLearningRate',1,...
    'biasesLearningRate',2,'filtersWeightDecay',1,'biasesWeightDecay',0);

net.layers{end+1} = struct('type', 'bnorm',...
                             'weights', {{ones(1024, 1, 'single'), zeros(1024, 1, 'single'), zeros(1024, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;

net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type','conv','weights',{{f*randn(1,1,1024,classNum,'single'), zeros(1,classNum,'single')}},...
    'stride', 1, 'pad', 0, 'filtersLearningRate', 1, ...
    'biasesLearningRate', 2, 'filtersWeightDecay', 1, 'biasesWeightDecay', 0);
% 
net.layers{end+1} = struct('type', 'bnorm',...
                             'weights', {{ones(classNum, 1, 'single'), zeros(classNum, 1, 'single'), zeros(classNum, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
% 
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'adaptive_pooling', 'pad', 0, 'stride', 1, 'method', 'avg') ;

net.layers{end+1} = struct('type', 'bnorm',...
                             'weights', {{ones(classNum, 1, 'single'), zeros(classNum, 1, 'single'), zeros(classNum, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;

net.layers{end+1}=struct('type', 'multi_label_loss') ;

[net, info] = cnn_train(net, imdb, @getBatch, opts.train, 'val', find(imdb.images.set==3)) ;


% -------------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% -------------------------------------------------------------------------
for i=1:size(batch,2)
    img=imdb.images.data(:,:,:,batch(i));
    img=single(img);
    im(:,:,:,i)=img;
    labels(i, :)=imdb.images.class(batch(i), :);
    labels(find(labels == -1)) = 0 ;
end


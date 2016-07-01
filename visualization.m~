%% Visualizing typical layers in the network
%% Note that I used some files from Bolei Zhou's source code
%% Created by Hong-Yu Zhou at 2016.6.03

run ./matconvnet-1.0-beta20/matlab/vl_setupnn
addpath ./matconvnet-1.0-beta20/matlab/
addpath ./matconvnet-1.0-beta20/examples/

load('./data/pretrained_model.mat') ;
net = vl_simplenn_move(net, 'gpu') ; % net on gpu is faster
load('imdb.mat') ;
ids = find(imdb.images.set==3) ;
net.layers(end) = [] ;
categories = {'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'}

for i = 1:size(ids,2)
    img_ = imdb.images.data(:,:,:,ids(i)) ;
    img = imresize(img_, [256 256]);
    curResult = im2double(img) ;
    curPrediction  =  '' ;
    res = vl_simplenn(net, gpuArray(single(img_))) ;
    [~, pred] = sort(squeeze(gather(res(end).x)), 'descend') ;
    index = find(imdb.images.class(ids(i), :)~=-1) ;
    % 35 36 37
    layer_35 = res(35).x ;
    for j = 1:numel(index)        
        activationMap = squeeze(gather(layer_35(:,:,index(j)))) ;
        activationMap = imresize(im2double(activationMap), [256 256]) ;
        curHeatMap = im2double(activationMap) ;
        
        curHeatMap = map2jpg(curHeatMap,[], 'jet') ;
        curHeatMap = im2double(img)*0.2+curHeatMap*0.7 ;
        curResult = [curResult ones(size(curHeatMap,1),8,3) curHeatMap] ;
        curPrediction = [curPrediction num2str(i) ' --gt'  num2str(index(j)) ':' categories{index(j)}] ;
    end
    figure,imshow(curResult);title(curPrediction)
    pause
end

close all

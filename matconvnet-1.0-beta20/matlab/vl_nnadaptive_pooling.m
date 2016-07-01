function [result,averagepooling] = vl_nnadaptive_pooling(x,layers,pad,stride,method,opts, cudnn,dzdy)

%% global averagepooling
if nargin<=7
    szData=size(x);
    poolH=szData(1);
    poolW=szData(2);
    %layers.pool=[poolH poolW];
    %global averagepooling
    averagepooling=[poolH poolW];
    result=vl_nnpool(x, averagepooling, ...
        'pad', pad, 'stride', stride, ...
        'method', method, ...
        cudnn);
else
    result=vl_nnpool(x, layers, single(dzdy),...
        'pad', pad, 'stride', stride, ...
        'method', method, ...
        cudnn);
    averagepooling=layers;
end

end


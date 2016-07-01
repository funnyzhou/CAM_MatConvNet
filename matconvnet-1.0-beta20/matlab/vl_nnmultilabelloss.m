function Y = vl_nnmultilabelloss(X,c,dzdy)
%VL_NNSOFTMAXLOSS CNN combined softmax and logistic loss.
%   **Deprecated: use `vl_nnloss` instead**
%
%   Y = VL_NNSOFTMAX(X, C) applies the softmax operator followed by
%   the logistic loss the data X. X has dimension H x W x D x N,
%   packing N arrays of W x H D-dimensional vectors.
%
%   C contains the class labels, which should be integers in the range
%   1 to D. C can be an array with either N elements or with dimensions
%   H x W x 1 x N dimensions. In the fist case, a given class label is
%   applied at all spatial locations; in the second case, different
%   class labels can be specified for different locations.
%
%   DZDX = VL_NNSOFTMAXLOSS(X, C, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;
y = [-1,1] ;
y = single(y) ;
classNum = size(c, 2) ;
trainSize = size(c, 1) ;
Y = gpuArray(single(0)) ;
if nargin<=2
    for i = 1:trainSize
        for j = 1:classNum
            if c(i,j) == 1
                Y = Y + log(1 + exp(-y(2)*X(1,1,j,i))) ;
            else
                Y = Y + log(1 + exp(-y(1)*X(1,1,j,i))) ;
            end
        end
        %Y=0;
    end
else
    Y = gpuArray(single(zeros(size(X)))) ;
    %y=single(y);
    for i = 1:trainSize
        for j=1:classNum
            if c(i,j) == 1
                Y(1,1,j,i) = dzdy*((-y(2)*exp(-y(2)*X(1,1,j,i)))/(1+exp(-y(2)*X(1,1,j,i))));
                %((-y(2)*exp(-y(2)*X(1,1,j,i)))/(1+exp(-y(2)*X(1,1,j,i))));
            else
                Y(1,1,j,i) = dzdy*((-y(1)*exp(-y(1)*X(1,1,j,i)))/(1+exp(-y(1)*X(1,1,j,i))));
                %((-y(1)*exp(-y(1)*X(1,1,j,i)))/(1+exp(-y(1)*X(1,1,j,i))));
            end
        end
    end
end

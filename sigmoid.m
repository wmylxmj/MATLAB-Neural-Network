function [ s ] = sigmoid( z )
%SIGMOID sigmoid
%   此处显示详细说明
s = (1 + exp(-z)).^(-1) ;
end


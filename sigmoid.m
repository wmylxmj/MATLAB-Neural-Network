function [ s ] = sigmoid( z )
%SIGMOID sigmoid
%   �˴���ʾ��ϸ˵��
s = (1 + exp(-z)).^(-1) ;
end


clear
clc

load('data.mat');

for i=1:200
    dot = X(i,:);
    if y(i,1)==1
        plot(dot(:,1),dot(:,2), 'o','Color','b');
        hold on;
    else
        plot(dot(:,1),dot(:,2), 'o','Color','r');
        hold on;
    end
end

hold on

% layers:2 20 7 1

X_train = X';
Y_train = y';
m = 211;
learning_rate = 1.2;

% init params
W1 = randn(20,2).*sqrt(2.0/2);
b1 = 0;
W2 = randn(7,20).*sqrt(2.0/20);
b2 = 0;
W3 = randn(1,7).*sqrt(2.0/7);
b3 = 0;

for epoch=1:10000
    Z1 = W1*X_train + b1;
    A1 = max(Z1,0);
    Z2 = W2*A1 + b2;
    A2 = max(Z2,0);
    Z3 = W3*A2 + b3;
    A3 = sigmoid(Z3);
    y_hat = A3;
    cost = (1.0/m).* -(Y_train * log(y_hat') + (1-Y_train) * log(1-y_hat'));
    disp(cost);
    % layer 3
    dA3 = -(Y_train./A3 - (1-Y_train)./(1-A3));
    dZ3 = A3 - Y_train;
    dW3 = (1.0/m) .* (dZ3*A2');
    db3 = (1.0/m) .* sum(dZ3, 2);
    % layer 2
    dA2 = W3' * dZ3;
    dZ2 = dA2.*(A2>0);
    dW2 = (1.0/m) .* (dZ2*A1');
    db2 = (1.0/m) .* sum(dZ2, 2);
    % layer 1
    dA1 = W2' * dZ2;
    dZ1 = dA1.*(A1>0);
    dW1 = (1.0/m) .* (dZ1*X_train');
    db1 = (1.0/m) .* sum(dZ1, 2);
    % update
    W1 = W1 - learning_rate .* dW1;
    b1 = b1 - learning_rate .* db1;
    W2 = W2 - learning_rate .* dW2;
    b2 = b2 - learning_rate .* db2;
    W3 = W3 - learning_rate .* dW3;
    b3 = b3 - learning_rate .* db3;
end

test_flag = 1;

if test_flag    
    for x1=-0.6:0.04:0.3
        for x2=-0.8:0.04:0.6
            x = [x1;x2];
            Z1 = W1*x + b1;
            A1 = max(Z1,0);
            Z2 = W2*A1 + b2;
            A2 = max(Z2,0);
            Z3 = W3*A2 + b3;
            A3 = sigmoid(Z3);
            y_hat = A3;
            if y_hat>=0.5
                plot(x1,x2, 'p','Color','b');
                hold on;
            else
                plot(x1,x2, 'p','Color','r');
                hold on;
            end
        end
    end
end
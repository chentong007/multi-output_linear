% Multiple output by linear classifier
% Prediction of energy consumption
load('D_build_tr');
load('D_build_te');

% Prepare training dataset
Xtr = D_build_tr(1:8,:);
Ytr = D_build_tr(9:10,:);

% Prepare test dataset
Xte = D_build_te(1:8,:);
Yte = D_build_te(9:10,:);

% Training dataset
x_hat = ones(640,9);
x_hat(1:640,1:8) = Xtr';
w_hat = inv(x_hat'*x_hat+0.01)*x_hat'*Ytr';
w = w_hat';
w_output = w_hat(1:8,:);
b = w(:,9);

% Testing dataset
y_test = w_output'*Xte+b*ones(1,128);

% Error
error_p = norm(Yte-y_test)/norm(Yte);
   
% Display prediction results: comprare with true labels
figure (1)
plot(Yte(1,:),'r');
hold on
plot(y_test(1,:),'b');
grid
title('Y true labels shown in red, Y predictions shown in blue');
xlabel('First set of data');

figure (2)
plot(Yte(2,:),'r');
hold on
plot(y_test(2,:),'b');
grid
title('Y true labels shown in red, Y predictions shown in blue');
xlabel('Second set of data');
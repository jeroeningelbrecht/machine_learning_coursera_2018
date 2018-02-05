function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_new = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

for col=1:size(X,2)
    predictions=X*theta;
    var_help = (predictions-y)*(X(:,col))';
    theta(:,col)=theta(:,col)-alpha/m*var_help;
endfor
    

%  for j=1:length(theta)
%    var_help=0;
%    for i=1:m
%      var_help = var_help + ((theta'*(X(i,:))'-y(i))*X(i,j));
%    endfor
%    theta_new(j)=theta(j) - alpha/m * var_help;
%  endfor

 % theta = theta_new;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

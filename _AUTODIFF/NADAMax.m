function [x,m,v,nIter] = NADAMax(grad,x,beta1,beta2,step_adam,epsilon,m,v,nIter)
%Dozat, T. (2016). Incorporating Nesterov Momentum into Adam.
%last line adapted from : https://github.com/saeidsoheily/gradient-descent-variants-optimization-algorithms/blob/master/nadam.py

%beta1 = 0.9;
%beta2 = 0.999;
%epsilon = eps(single(1));

m = beta1.*m + (1-beta1).*grad;
mHat = m./(1 - beta1^nIter);

v = max(beta2.*v, abs(grad));
x = x + step_adam./(v+epsilon).*(beta1 * mHat + ((1 - beta1) * grad / (1 - beta1.^nIter)));
end


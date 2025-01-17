function [x,m,v,nIter] = NADAM(grad,x,beta1,beta2,step_adam,epsilon,m,v,nIter)
%Dozat, T. (2016). Incorporating Nesterov Momentum into Adam.
%last line adapted from : https://github.com/saeidsoheily/gradient-descent-variants-optimization-algorithms/blob/master/nadam.py

%beta1 = 0.9;
%beta2 = 0.999;
%epsilon = eps(single(1));



m = beta1.*m + (1 - beta1) .* grad;
v = beta2.*v + (1 - beta2) .* (grad.^2);

mHat = m./(1 - beta1^nIter);
vHat = v./(1 - beta2^nIter);

%x = x + step_adam*mHat./(sqrt(vHat) + epsilon);
x = x + step_adam./(sqrt(vHat) + epsilon).*(beta1 * mHat + ((1 - beta1) * grad / (1 - beta1.^nIter)));
nIter=nIter+1;
end


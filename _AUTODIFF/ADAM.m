function [x,m,v,nIter] = ADAM(grad,x,beta1,beta2,step_adam,epsilon,m,v,nIter)

%beta1 = 0.9;
%beta2 = 0.999;
%epsilon = eps(single(1));


m = beta1.*m + (1 - beta1) .* grad;
v = beta2.*v + (1 - beta2) .* (grad.^2);

mHat = m./(1 - beta1^nIter);
vHat = v./(1 - beta2^nIter);

x = x + step_adam*mHat./(sqrt(vHat) + epsilon);
nIter=nIter+1;
end


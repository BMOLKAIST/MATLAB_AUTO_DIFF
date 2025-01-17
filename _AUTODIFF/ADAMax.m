function [x,m,v,nIter] = ADAMax(grad,x,beta1,beta2,step_adam,epsilon,m,v,nIter)

%beta1 = 0.9;
%beta2 = 0.999;
%epsilon = eps(single(1));

m = beta1.*m + (1-beta1).*grad;
mHat = m./(1 - beta1^nIter);

v = max(beta2.*v, abs(grad));
x = x + step_adam.*mHat./(v+epsilon);
end


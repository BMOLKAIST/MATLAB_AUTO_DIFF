function [y,gradient] = adiff(f, x)
%% calculate the gradient dy=df/dconj(x)
    if nargout == 1
        y = f(x);
    else
        dy = []; %dx/dconj(x)=0 -> [] empty to avoid computations
        dy_c= 1; %dx/dx=1
        x = ADNode(x);
        y = f(x);
        [gradient] = y.backprop(dy,dy_c);
        y = y.value;
        
    end
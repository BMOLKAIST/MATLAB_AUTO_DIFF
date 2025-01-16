function [y,gradient] = adiff_numerical(f, x,delta)

y=f(x);
gradient=0.*x;

for ii=1:length(x(:))
    if ~mod(ii,round(length(x(:))/100))
        display(['Numerical diff : ' num2str(100*ii./length(x(:))) '/100']);
    end
    new_x=x;
    new_x(ii)=new_x(ii)+1.*delta;
    cost_1=f(new_x);
    gradient(ii)=(cost_1-y)/(2*delta);
    new_x=x;
    new_x(ii)=new_x(ii)+1i.*delta;
    cost_1=f(new_x);
    gradient(ii)=gradient(ii)+1i.*(cost_1-y)/(2*delta);
end
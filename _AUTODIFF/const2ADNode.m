function y =const2ADNode(x,ref)
if isa(ref,'ADNode')
    y=ADNode(x, ref.root, @(y) 0);
else
    y=x;
end
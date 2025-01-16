function out=s_ifftn(in)
out=sqrt(prod(size(in))).*fftshift(ifftn(ifftshift(in)));
end
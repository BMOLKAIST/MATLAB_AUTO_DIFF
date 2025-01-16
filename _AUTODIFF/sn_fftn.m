function out=s_fftn(in)
out=(1/sqrt(prod(size(in)))).*fftshift(fftn(ifftshift(in)));
end
function out=sn_ifft2(in)
out=sqrt(size(in,1)*size(in,2)).*fftshift(ifft2(ifftshift(in)));
end
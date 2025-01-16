function out=sn_ifft3(in)
out=sqrt(size(in,1)*size(in,2)*size(in,2)).*fftshift(ifft3(ifftshift(in)));
end
function out=sn_ifft(in,n)
out=sqrt(size(in,n)).*fftshift(ifft(ifftshift(in,n),[],n),n);
end
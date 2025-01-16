function out=s_ifft(in,n)
out=fftshift(ifft(ifftshift(in),[],n));
end
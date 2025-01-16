function out=sn_fft(in,n)
out=(1/sqrt(size(in,n))).*fftshift(fft(ifftshift(in,n),[],n),n);
end

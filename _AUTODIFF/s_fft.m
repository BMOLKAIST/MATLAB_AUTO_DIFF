function out=s_fft(in,n)
out=fftshift(fft(ifftshift(in),[],n));
end

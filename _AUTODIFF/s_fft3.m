function out=s_fft3(in)
out=fftshift(fft3(ifftshift(in)));
end
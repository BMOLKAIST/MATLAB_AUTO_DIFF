function out=s_fft2(in)
out=fftshift(fft2(ifftshift(in)));
end

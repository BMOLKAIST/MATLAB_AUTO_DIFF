function out=sn_fft2(in)
out=(1/sqrt(size(in,1)*size(in,2))).*fftshift(fft2(ifftshift(in)));
end

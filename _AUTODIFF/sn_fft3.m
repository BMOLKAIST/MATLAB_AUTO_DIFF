function out=sn_fft3(in)
out=(1/sqrt(size(in,1)*size(in,2)*size(in,3))).*fftshift(fft3(ifftshift(in)));
end
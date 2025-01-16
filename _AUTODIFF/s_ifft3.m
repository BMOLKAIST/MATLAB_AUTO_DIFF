function out=s_ifft3(in)
out=fftshift(ifft3(ifftshift(in)));
end
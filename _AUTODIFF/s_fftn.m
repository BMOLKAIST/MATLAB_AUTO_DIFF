function out=s_fftn(in)
out=fftshift(fftn(ifftshift(in)));
end
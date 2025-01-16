function out=s_ifftn(in)
out=fftshift(ifftn(ifftshift(in)));
end
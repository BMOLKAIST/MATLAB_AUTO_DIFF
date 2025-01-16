function img=imshearotate(img,angle)
pixel=angle*size(img,1)/360*2*pi;
img=gpuArray(single(img));
coo1=(1:size(img,1))-floor(size(img,1)/2)-1;coo1=coo1./size(img,1);
coo1=gpuArray(single(reshape(coo1,[],1)));
coo2=(1:size(img,2))-floor(size(img,2)/2)-1;coo2=coo2./size(img,2);
coo2=gpuArray(single(reshape(coo2,1,[])));
kernel=exp(-1i.*2.*pi.*(pixel*coo1*coo2));
img=s_ifft(s_fft(img,1).*kernel,1);
img=s_ifft(s_fft(img,2).*conj(kernel),2);
%{
for ii=1:size(img(:,:,:),3)
    img(:,:,ii)=s_ifft(s_fft(img(:,:,ii),1).*kernel,1);
    img(:,:,ii)=s_ifft(s_fft(img(:,:,ii),2).*conj(kernel),2);
end
%}
end

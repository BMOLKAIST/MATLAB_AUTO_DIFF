%to test 3D recon the camera man is extended 200 pixel in the z direction
img=gpuArray(single(imread('cameraman.tif')));
img2=imresize(gpuArray(single(imread('moon.tif'))),size(img));
img=cat(3,img,1i.*img2,img2+1i.*img);
%img=sino1;
angle=0:0.5:179.9999;
over_padd=1;
padd_sz=ceil(size(img).*[sqrt(2) sqrt(2) 1]);
img_padd=padd_crop_to_fit(img,padd_sz);
slice=radon_adiff(img_padd,angle,over_padd,false);
%figure; imagesc(squeeze(slice(:,1,:))) axis image;
figure; sliceViewer(gather(permute(real(slice(:,:,:)),[1 3 2]))); axis image;
%%
%slice=radon(img,angle);slice=reshape(slice,size(slice,1),1,[]);
res=iradon_adiff(slice,angle,over_padd,true);
res=padd_crop_to_fit(res,size(img));
%figure; imagesc(real(squeeze(res(:,:,1)))); axis image;
figure; sliceViewer(gather(real(res(:,:,:)))); axis image;
figure; sliceViewer(gather(imag(res(:,:,:)))); axis image;

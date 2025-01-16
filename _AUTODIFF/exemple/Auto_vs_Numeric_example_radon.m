addpath('C:\HERVE\_AUTODIFF')

angles=0:10:179.9999;
over_padd=1;

img=imresize(gpuArray(single(imread('cameraman.tif'))),0.1);
img2=imresize(gpuArray(single(imread('moon.tif'))),size(img));
img=img+1i.*img2;
slice=padd_crop_to_fit(radon_adiff(img,angles,over_padd,true),[size(img,1).*sqrt(2) size(img,3) length(angles(:))]);

curr_slice=0.*slice;
curr_img=0.*img;

cost_function_slice=@(slice) mean(abs2(padd_crop_to_fit(iradon_adiff(slice,angles,over_padd,true),size(img))-img),'all');
cost_function_img=@(img) mean(abs2(radon_adiff(padd_crop_to_fit(img,[size(slice,1) size(slice,1) size(slice,2)]),angles,over_padd,false)-slice),'all');


%cost_function=cost_function_slice; curr_x=curr_slice;d_num=0.1;
cost_function=cost_function_img; curr_x=curr_img;d_num=10;


figure; imagesc(real(img)); axis image;

display('Executing automatic differenciation');

tic
[val grad]=adiff(cost_function, curr_x);
toc
figure;
subplot(2,2,1); imagesc(real(grad(:,:))); axis image; title('Automatic differentiation (real)');
subplot(2,2,2); imagesc(imag(grad(:,:))); axis image; title('Automatic differentiation (imag)');

drawnow

display('Executing numerical differenciation');

tic
[val_numerical grad_numerical]=adiff_numerical(cost_function, curr_x, d_num);
toc

subplot(2,2,3); imagesc(real(grad_numerical(:,:))); axis image; title('Numerical differentiation (real)');
subplot(2,2,4); imagesc(imag(grad_numerical(:,:))); axis image; title('Numerical differentiation (imag)');

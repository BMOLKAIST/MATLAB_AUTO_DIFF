addpath('C:\HERVE\_AUTODIFF')

img=imresize(imread("cameraman.tif"),0.2);
img2=imread('moon.tif');img2=imresize(img2,size(img));

field=exp(1i.*single(img)/1000+single(img2)/1000);


co1=((1:size(field,1))-floor(size(field,1)/2)-1)/size(field,1);co1=reshape(co1,[],1);
co2=((1:size(field,2))-floor(size(field,2)/2)-1)/size(field,2);co2=reshape(co2,1,[]);

shift=reshape([-8 8],1,1,[]);

propagation_kernel=single(exp(1i.*(co1.^2+co2.^2).*shift));

experiment=@(complex_field) abs2(sn_ifft2(propagation_kernel.*(sn_fft2(exp(complex_field))).*propagation_kernel ));

recorded_intensity=experiment(field);

cost_function=@(test_field) mean(abs2(experiment(test_field)-recorded_intensity),'all');
%cost_function=@(test_field) mean(abs2(test_field-recorded_intensity),'all');


figure; imagesc(recorded_intensity(:,:)); axis image;

curr_x=exp(1i.*single(img)./255);

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
[val_numerical grad_numerical]=adiff_numerical(cost_function, curr_x, 0.01);
toc

subplot(2,2,3); imagesc(real(grad_numerical(:,:))); axis image; title('Numerical differentiation (real)');
subplot(2,2,4); imagesc(imag(grad_numerical(:,:))); axis image; title('Numerical differentiation (imag)');

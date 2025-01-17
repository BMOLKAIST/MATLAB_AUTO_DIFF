addpath('C:\HERVE\_AUTODIFF')

%solves the multi plane phase retrieval problem 
%set the ground truth
img=imresize(imread("cameraman.tif"),1);
img2=imread('moon.tif');img2=imresize(img2,size(img));
field=gpuArray(single(exp(1i.*single(img)/100+single(img2)/1000)));
%coordinates
co1=((1:size(field,1))-floor(size(field,1)/2)-1)/size(field,1);co1=reshape(co1,[],1);
co2=((1:size(field,2))-floor(size(field,2)/2)-1)/size(field,2);co2=reshape(co2,1,[]);
%camera shifting distance
shift=reshape([100 150 200],1,1,[]);
%propagation kernel
propagation_kernel=gpuArray(single(exp(1i.*(co1.^2+co2.^2).*shift)));
%experimental image formation
experiment=@(complex_field) abs2(sn_ifft2(sn_fft2(complex_field).*propagation_kernel));
recorded_intensity=experiment(field);
cost_function=@(test_field) sum(abs2(sqrt(experiment(real(test_field(:,:,1)).*exp(1i.*real(test_field(:,:,2)))))-sqrt(recorded_intensity)),'all');
%cost_function=@(test_field) sum(abs2(sqrt(experiment(exp(test_field)))-sqrt(recorded_intensity)),'all');


figure; imagesc(recorded_intensity(:,:)); axis image;

%% FISTA optimization for inversion
curr_x=zeros([size(field) 2],'single','gpuArray');
curr_x(:,:,1)=1;

%FISTA
t_np=1;
x_n=curr_x;
y_n=curr_x;
step=-0.1;
itt_num=2000;
%ADAM
x_adam=curr_x;
m=0;
v=0;
beta1 = 0.9;
beta2 = 0.999;
step_adam=-0.1;
epsilon = eps(single(1));
nIter=1;

cost=zeros(itt_num,1,'single');
cost_adam=zeros(itt_num,1,'single');
figure;
tic;
for itt=1:itt_num
    [val, grad]=adiff(cost_function,y_n);
    [val_adam, grad_adam]=adiff(cost_function,x_adam);
    
    [y_n,x_n,t_np] = FISTA(step.*grad,y_n,x_n,t_np);
    [x_adam,m,v,nIter] = ADAM(grad_adam,x_adam,beta1,beta2,step_adam,epsilon,m,v,nIter);
    %[x_adam,m,v,nIter] = ADAMax(grad_adam,x_adam,beta1,beta2,step_adam,epsilon,m,v,nIter);
    
    cost(itt)=val;
    cost_adam(itt)=val_adam;

    if mod(itt,100)==0 || itt==itt_num
        subplot(4,2,1);imagesc(x_n(:,:,1)); axis image; title('amplitude optimized FISTA')
        subplot(4,2,2);imagesc(x_n(:,:,2)); axis image; title('phase optimized FISTA')
        
        subplot(4,2,3);imagesc(x_adam(:,:,1)); axis image; title('amplitude optimized ADAM')
        subplot(4,2,4);imagesc(x_adam(:,:,2)); axis image; title('phase optimized ADAM')
        
        subplot(4,2,5);imagesc(abs(real(log(field)))); axis image; title('amplitude ground truth');
        subplot(4,2,6);imagesc(imag(log(field))); axis image; title('phase ground truth');
        
        subplot(4,2,7);semilogy(cost(1:itt)); title('Cost function evolution FISTA');
        subplot(4,2,8);semilogy(cost_adam(1:itt)); title('Cost function evolution ADAM');
        drawnow;
    end

end
toc;


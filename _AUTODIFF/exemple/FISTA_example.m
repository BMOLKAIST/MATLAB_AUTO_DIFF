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

t_np=1;
x_n=curr_x;
itt_num=2000;

cost=zeros(itt_num,1,'single');
figure;
tic;
for itt=1:itt_num


    [val grad]=adiff(cost_function, curr_x);
    [curr_x,x_n,t_np] = FISTA(-0.1.*grad,curr_x,x_n,t_np);
    
    cost(itt)=val;

    if mod(itt,100)==0 || itt==itt_num
        subplot(3,2,1);imagesc(curr_x(:,:,1)); axis image; title('amplitude optimized')
        subplot(3,2,2);imagesc(curr_x(:,:,2)); axis image; title('phase optimized')
        
        subplot(3,2,3);imagesc(abs(real(log(field)))); axis image; title('amplitude ground truth');
        subplot(3,2,4);imagesc(imag(log(field))); axis image; title('phase ground truth');
        
        subplot(3,1,3);plot(cost(1:itt)); title('Cost function evolution');
        drawnow;
    end

end
toc;


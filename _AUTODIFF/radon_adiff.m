function [projections]=radon_adiff(res,angles,over_padd,use_filter)
%for_adiff_trans is a variable for autodifferentiation dont use it in normal use 

if isempty(res)
    
    projections=[];
    return;
end
real_flag=isreal(res);
use_count=false;
if size(res,1)~=size(res,2)
    error('matrix should be square in the two first dims');
end
padding=round(size(res,1)*over_padd);
res=padarray(res,[padding padding 0],0,'both');


side_sz=size(res,1);
side_3D=size(res,3);
center_sz=(floor(side_sz/2)+1);
c1=(1:side_sz)-center_sz;
if use_filter
    filter=(abs(c1))';
else
    filter=1;
end

res=sn_ifft2(res);
res=reshape(res,side_sz*side_sz,[]);

projections=zeros(side_sz,side_3D,length(angles(:)),'single','gpuArray');
projections=const2ADNode(projections,res); % to enable autodiffarentiation (in fact redundent since hard implemented in ADNode)

if use_count
count=zeros(side_sz,side_sz,1,'single','gpuArray');
end

for ii=1:length(angles(:))
    angle_rad=deg2rad(angles(ii));
    d1=gpuArray((mod(round(-c1.*sin(angle_rad)+center_sz)'-1,side_sz)+1));
    d2=gpuArray((mod(round(c1.*cos(angle_rad)+center_sz)'-1,side_sz)+1));
    

    subs=(d1)+(d2-1).*side_sz;
    
    projections((1:side_sz),:,ii)=res(subs,:);
    
    if use_count
        count(uniquevals)=count(uniquevals)+filter(idxUnique);
    end
    
end
if use_count
    count=count+res.*0;
    res(count>0)=res((count>0))./count((count>0));
end

projections=projections.*filter;
projections=sn_fft(projections,1);

%max_val=max(abs(res(:)));

%res=res./max_val;

%figure; imagesc(real(squeeze(projections)))

projections=projections(1+padding:end-padding,:,:);

if real_flag
    projections=real(projections);
end
end
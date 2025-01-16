function [projections]=radon_adiff(res,angles,over_padd,use_filter)
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

res=sn_fft2(res);

projections=zeros(side_sz,side_3D,length(angles(:)),'single','gpuArray');
if use_count
count=zeros(side_sz,side_sz,1,'single','gpuArray');
end

for ii=1:length(angles(:))
    angle_rad=deg2rad(angles(ii));
    d1=((mod(round(-c1.*sin(angle_rad)+center_sz)'-1,side_sz)+1));
    d2=((mod(round(c1.*cos(angle_rad)+center_sz)'-1,side_sz)+1));
    d3=(((1:side_3D)-1));
    field_info=projections(:,:,ii);

    subs=gpuArray(d1+(d2-1).*side_sz);
    %[uniquevals,idxUnique,~] = unique(subs);
    %res(subs+d3.*side_sz.*side_sz);
    %size((1:side_sz))
    %size(d3)
    
    
    field_info((1:side_sz)'+d3.*size(field_info,1))=res(subs+d3.*side_sz.*side_sz);
    
    if use_count
        count(uniquevals)=count(uniquevals)+filter(idxUnique);
    end
    
    projections(:,:,ii)=field_info;
end
if use_count
    count=count+res.*0;
    res(count>0)=res((count>0))./count((count>0));
end

projections=projections.*filter;
projections=sn_ifft(projections,1);


%max_val=max(abs(res(:)));

%res=res./max_val;

%figure; imagesc(real(squeeze(projections)))

projections=projections(1+padding:end-padding,:,:);

end
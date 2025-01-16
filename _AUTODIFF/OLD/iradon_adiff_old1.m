function [res]=iradon_adiff(projections,angles,over_padd,use_filter)
use_count=false;

padding=round(size(projections,1)*over_padd);
projections=padarray(projections,padding,0,'both');


side_sz=size(projections,1);
side_3D=size(projections,2);
center_sz=(floor(side_sz/2)+1);
c1=(1:side_sz)-center_sz;
if use_filter
    filter=(abs(c1))';
else
    filter=1;
end

projections=sn_fft(projections,1);
projections=projections.*filter;

res=zeros(side_sz,side_sz,side_3D,'single','gpuArray');
if use_count
count=zeros(side_sz,side_sz,1,'single','gpuArray');
end

for ii=1:length(angles(:))
    angle_rad=deg2rad(angles(ii));
    %d1=round(-c1.*sin(angle_rad)+center_sz)';
    %d2=round(c1.*cos(angle_rad)+center_sz)';
    d1=mod(round(-c1.*sin(angle_rad)+center_sz)'-1,side_sz)+1;
    d2=mod(round(c1.*cos(angle_rad)+center_sz)'-1,side_sz)+1;
    d3=(1:side_3D)-1;
    field_info=projections(:,:,ii);

    subs=gpuArray(d1+(d2-1).*side_sz);
    for mm=1:2
    %[uniquevals,idxUnique,~] = unique(subs); error('need to acound for non unique ones')
    idxUnique=(mm:2:length(subs))';
    uniquevals=subs(idxUnique);
    res(uniquevals+d3.*side_sz.*side_sz)=res(uniquevals+d3.*side_sz.*side_sz)+field_info(idxUnique+d3.*size(field_info,1));
    end
    if use_count
        count(uniquevals)=count(uniquevals)+filter(idxUnique);
    end
end
if use_count
    count=count+res.*0;
    res(count>0)=res((count>0))./count((count>0));
end


res=sn_ifft2(res);


%max_val=max(abs(res(:)));

%res=res./max_val;

res=res(1+padding:end-padding,1+padding:end-padding,:);

end
function [res]=iradon_adiff(projections,angles,over_padd,use_filter)

if isempty(projections)
    res=[];
    return;
end
real_flag=isreal(projections);

use_count=false;

padding=round(size(projections,1)*over_padd);
%projections=padarray(projections,[padding 0 0],0,'both');
projections=padd_crop_to_fit(projections,size(projections)+2*[padding,0,0]);


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

res=zeros(side_sz*side_sz,side_3D,'single','gpuArray');
res=const2ADNode(res,projections);% to enable autodiffarentiation (in fact redundent since hard implemented in ADNode)

if use_count
    count=zeros(side_sz,side_sz,1,'single','gpuArray');
end

for ii=1:length(angles(:))
    angle_rad=deg2rad(angles(ii));

    d1=((mod(round(-c1.*sin(angle_rad)+center_sz)'-1,side_sz)+1));
    d2=((mod(round(c1.*cos(angle_rad)+center_sz)'-1,side_sz)+1));

    subs=(d1+(d2-1).*side_sz);

    for mm=1:2
        idxUnique=(mm:2:length(subs))';
        uniquevals=subs((idxUnique));

        res(uniquevals,:)=projections(idxUnique,:,ii)+res(uniquevals,:);
    end

    if use_count
        count(uniquevals)=count(uniquevals)+filter(idxUnique);
    end
end
if use_count
    count=count+res.*0;
    res(count>0)=res((count>0))./count((count>0));
end

res=reshape(res,[side_sz,side_sz,side_3D]);

res=sn_ifft2(res);


%max_val=max(abs(res(:)));

%res=res./max_val;

%res=res(1+padding:end-padding,1+padding:end-padding,:);

sz_out=size(res);
for pp=1:2
    sz_out(pp)=sz_out(pp)-2*padding;
end
res=padd_crop_to_fit(res,sz_out);

if real_flag
    res=real(res);
end
end
function img_out=padd_crop_to_fit(img_out,sizing)
if isempty(img_out)
    return;
end
%crop orpadd an image while keep the DC at the same place for fourier transforms
sz1 = size(img_out);
sz2 = sizing;
while length(sz1)<length(sz2)
    sz1(end+1)=1;
end
%{
while length(sz2)<length(sz1)
    sz2(end+1)=1;
end
%}
subs=cell(length(sz1),1);
subs(:)={':'};
pre_pad_sz=sz2.*0;
post_pad_sz=sz2.*0;
for tt=1:length(sz2)
    if sz1(tt)<sz2(tt)
        floor_sz=floor((sz2(tt)-sz1(tt))/2);
        ceil_sz=ceil((sz2(tt)-sz1(tt))/2);
        if mod(sz1(tt),2)==0
            pre_pad_sz(tt)=floor_sz;
            post_pad_sz(tt)=ceil_sz;
        else
            pre_pad_sz(tt)=ceil_sz;
            post_pad_sz(tt)=floor_sz;
        end
    end
    if sz1(tt)>sz2(tt)
        floor_sz=floor((sz1(tt)-sz2(tt))/2);
        ceil_sz=ceil((sz1(tt)-sz2(tt))/2);
        if mod(sz1(tt),2)==0
            subs{tt}=1+ceil_sz:sz1(tt)-floor_sz;
        else
            subs{tt}=1+floor_sz:sz1(tt)-ceil_sz;
        end
    end
end
img_out = padarray(img_out,pre_pad_sz,'pre');
img_out = padarray(img_out,post_pad_sz,'post');
%subs
img_out = subsref(img_out,substruct('()',subs));




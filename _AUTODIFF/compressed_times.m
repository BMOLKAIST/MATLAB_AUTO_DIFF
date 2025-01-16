function [var]=compressed_times(var,scalars,siz_list)
curr_pos=1;
for ii=1:numel(siz_list)
    curr_size=prod(siz_list{ii},'all');
    var(curr_pos:curr_pos+curr_size-1)=var(curr_pos:curr_pos+curr_size-1).*scalars(ii);
    curr_pos=curr_pos+curr_size;
end
end
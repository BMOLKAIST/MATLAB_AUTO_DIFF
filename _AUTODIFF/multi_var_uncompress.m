function [varargout]=multi_var_uncompress(var,siz_list)
curr_pos=1;
for ii=1:nargout
    curr_size=prod(siz_list{ii},'all');
    varargout{ii}=reshape(var(curr_pos:curr_pos+curr_size-1),siz_list{ii});
    curr_pos=curr_pos+curr_size;
end
end
function [var,siz_list]= multi_var_compress(varargin)
siz_list=cell(nargin,1);

sz_tot=0;
for ii=1:nargin
    sz_tot=sz_tot+prod(size(varargin{ii}),'all');
end
var=zeros(sz_tot,1,'like',varargin{1});
curr_pos=1;
for ii=1:nargin
    curr_size=prod(size(varargin{ii}),'all');
    siz_list{ii}=size(varargin{ii});
    var(curr_pos:curr_pos+curr_size-1)=varargin{ii};
    curr_pos=curr_pos+curr_size;
end

end


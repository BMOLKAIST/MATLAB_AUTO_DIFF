function res=ifft3(matt)

res=matt;
matt=reshape(matt,size(matt,1),size(matt,2),size(matt,3),[]);
for ii=1:size(matt,4)
   res(:,:,:,ii)=ifftn(matt(:,:,:,ii));
end
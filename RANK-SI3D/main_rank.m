clc;
clear all;
warning off;
addpath('liblinear/matlab');
datapath='./front_depth/';      
savepath= './front_depth_rank/';  
list=dir(datapath);
tic
for ii=3:length(list)   
    filename = list(ii).name;
    full_dir=fullfile(datapath,filename);  
    data=load(full_dir);  
    data=data.depthmap;
     if str2double(filename(2:3))>30  
          continue;   
     end
     
     save_subfolder = fullfile(savepath, filename(1:end-4));
     if ~exist(save_subfolder)
         mkdir(save_subfolder);
     end

     savename=fullfile(save_subfolder,[filename(1:end-4), '.jpg']);  
      if exist(savename,'file')==2 
          continue;                
      end

     numberofframes = size(data,3);
     wd=320;  
     ht=240; 
     Depths=zeros(ht,wd,1,numberofframes); 
      for iq = 1 : numberofframes

            depthmap = data;
            depth=imresize(depthmap,[240,320]); 
            Depths(:,:,1,iq)=depth(:,:,iq);
      end
    
     [zWF] = depth_GetDynamicImages4(Depths);
     for s = 1: size(zWF,4)
         final = zWF(:, :, :, s);
         imwrite(final, [savename(1:end-4), '_',num2str(s),'.jpg']);
     end
     

end
toc

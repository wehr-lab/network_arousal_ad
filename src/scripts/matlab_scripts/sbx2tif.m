function sbx2tif(fname,varargin)

% sbx2tif
% Generates tif file from sbx files
% Argument is the number of frames to convert
% If no argument is passed the whole file is converted

z = sbxread(fname,1,1);
global info;

if(nargin>1)
    N = min(varargin{1},info.max_idx);
else
    N = info.max_idx;
end

k = 0; %SFM 9/3/24: Set k to starting frame + 1 for the plane
done = 0;
while(~done && k<=N)
    try
        q = sbxread(fname,k,1);
        q = squeeze(q(1,:,142:653)); %SFM 9/3/24: You can crop the dead columns on the side of a recording by specifying columns in the 3rd dimension here
        if(k==1)
            imwrite(q,[fname '.tif'],'tif');
        else
            imwrite(q,[fname '.tif'],'tif','writemode','append');
        end
    catch
        done = 1;
    end
    k = k+1;
end
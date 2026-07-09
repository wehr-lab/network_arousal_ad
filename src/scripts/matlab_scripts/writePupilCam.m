function [] = writePupilCam(varargin)

% Convert an *_eye.mat data structure into a MP4 Video
% Pass a single 2P directory or a list of directories. To batch process, pass a mouse directory (i.e., ~/JarascopeData/[mouse ID])

if length(varargin) == 1
    datadir = convertCharsToStrings(varargin{1});
elseif length(varargin) > 1
    for i = 1:length(varargin)
        datadir(i) = convertCharsToStrings(varargin{i});
    end
else
    error('No directory(s) passed to plot!')
end

dirs_to_convert = 0;
for i = 1:length(datadir)
    pathparts = strsplit(datadir{i}, '/');
    pathparts = rmmissing(pathparts);
    if length(pathparts) == 5
        allSessions = dir(fullfile(datadir{i}, '*-*'));
        for iSess = 1:length(allSessions)
            pupilFile = dir(fullfile(allSessions(iSess).folder, allSessions(iSess).name, '*_eye*'));
            if ~isempty(pupilFile) && length(pupilFile) == 1
                dirs_to_convert = dirs_to_convert + 1;
                newDatadir{dirs_to_convert} = pupilFile.folder;
            end
        end
    end
end
if exist('newDatadir') == 1
    datadir = newDatadir;
end

for iDir = 1:length(datadir)
    fullTerm = fullfile(datadir{iDir}, '*_eye.mat');
    matDir = dir(fullTerm);
    if isempty(matDir)
        error('Could not find pupil camera data in this directory')
    end
    tempstr = strsplit(matDir.name, '.');
    savename = fullfile(matDir.folder, tempstr{1});

    pupilCam = fullfile(matDir.folder, matDir.name);

    load(pupilCam);

    data = data(:,:,1,(24:end-8)); %These numbers are consistent for a given requested FPS

    % Uncomment this loop to determine what are the first and last frames with pupil data, and crop the video accordingly (i.e. if you change requested FPS from 24)
    % for iFrame = 1:30
    %     iFrameEnd = 30-iFrame; 
    %     endIndex = size(data, 4) - iFrameEnd;
    %     meanIntensityInitial(iFrame) = mean(data(:,:,1,iFrame),'all');
    %     meanIntensityEnd(iFrame) = mean(data(:,:,1,endIndex),'all');
    % end
    % recStartLog = meanIntensityInitial >= 50; 
    % startLog = find(recStartLog);
    % recStopLog = meanIntensityEnd >= 50; 
    % endLog = find(recStopLog);
    % endFrame = 30-endLog(end);
    % data = data(:,:,1,(startLog(1):end-endFrame));

    videoObj = VideoWriter(savename, 'MPEG-4');
    videoObj.FrameRate = 17.64; % Should this be hard-coded or determined via file information? 17.64 is the calculated FPS when requesting 24
    open(videoObj);
    writeVideo(videoObj, data);
    close(videoObj);
end

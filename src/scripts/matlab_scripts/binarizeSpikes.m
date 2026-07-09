function [] = binarizeSpikes(varargin)

% Convert a Suite2P data directory's deconvolved spks into a binary raster
% Input should be a two-photon data directory, OR the individual matched_suite2p session
% Optional argument: minimum spiking threshold smin (defaults to 0.5)
% smin * maxvalue/cell = spike binarize threshold


if ~isempty(varargin)
    datadir = convertCharsToStrings(varargin{1});
else
    error('Gotta pass a directory to plot!')
end
if length(varargin) > 1
    smin = varargin(2);
else
    smin = 0.5;
end

datapathparts = strsplit(datadir, '/');
mouseID = datapathparts{6}; 
sessionID = datapathparts{end};
basedir = '/Volumes/Projects/2P5XFAD/JarascopeData/'; % full data directory path to build subsequent filepaths from
FallFullPath = fullfile(datadir, '/suite2p/plane0/Fall.mat');
FallPath = fullfile(datadir, '/suite2p/plane0/');

FallDir = dir(FallFullPath);
if isempty(FallDir)
    Output = dir(fullfile(FallPath, '*.npy'));
    spks = readNPY(fullfile(FallPath, 'spks.npy'));
else
    load(FallFullPath)
end

for iCell = 1:size(spks,1)
    maxResp = max(spks(iCell,:));
    rast = spks(iCell,:) >= (maxResp * smin);
    binSpikes(iCell, :) = rast;
end

savename = fullfile(FallPath, 'binSpikes.mat');
save(savename, 'binSpikes');
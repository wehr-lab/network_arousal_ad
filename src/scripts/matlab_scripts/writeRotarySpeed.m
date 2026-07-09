function [] = writeRotarySpeed(varargin)

% pass a data directory to convert a *_quadrature.mat into running speed (cm/s)

if ~isempty(varargin)
    datadir = convertCharsToStrings(varargin{1});
end

fullTerm = fullfile(datadir, '*_quadrature.mat');
matDir = dir(fullTerm);
if isempty(matDir)
    error('Could not find locomotion data in this directory')
end
tempstr = strsplit(matDir.name, '_');
savename = fullfile(matDir.folder, strcat(tempstr{1}, '_', tempstr{2}, '_', tempstr{3}, '_locomotion.mat'));

quadFile = fullfile(matDir.folder, matDir.name);

load(quadFile);
frameDur = 0.0646;
quad_data = double(quad_data);
speed = zeros([1, length(quad_data)]);

for iFrame = 2:length(quad_data)
    speed(iFrame) = (0.005469 * (quad_data(iFrame) - quad_data(iFrame - 1)))/0.0646;
    %cm/s
    
end

save(savename, 'speed');

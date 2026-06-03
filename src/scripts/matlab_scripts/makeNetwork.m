function [] = makeNetwork(varargin)

if ~isempty(varargin)
    datadir = convertCharsToStrings(varargin{1});
else
    error('Need to input directory you want to construct network from') % enter directory to plot (if none explicitly passed)
end
filepathparts = strsplit(datadir, '/'); mouseID = filepathparts{6}; sessionID = filepathparts{end};
behaviorMAT = dir(fullfile(datadir, 'wehr*.mat'));
load(fullfile(datadir, behaviorMAT(1).name))
FallPath = fullfile(datadir, '/suite2p/plane0/Fall.mat');
load(FallPath)
iscellList = load(FallPath, 'iscell');
iscellList = iscellList.iscell;
clear iscell

iscellLog = logical(iscellList(:, 1)); iscellThresh = iscellList(:, 2);
% Uncomment and enter a threshold value (0-1) to use Suite2P's likelihood value to select good cells
% S2Pthresh = 0.95; % Suite2P likelihood threshold to use
% iscellLog = iscellThresh >= S2Pthresh;
cellsToPlot = F(iscellLog, :);
neucellsToPlot = Fneu(iscellLog, :);

corrScalar = 0.7;
cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);

minVal = min(cellToPlotCorr, [], 'all');
if minVal < 1
    cellsShifted = cellsToPlotCorr(:,:) + (1 - minVal);
end


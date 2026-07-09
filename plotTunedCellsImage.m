function [fig1, fig2] = plotTunedCellsImage(varargin)

plotDir = convertCharsToStrings(varargin{1});
if length(varargin) >= 2
    tunedList = varargin{2};
    tunedList = sort(tunedList, 'ascend'); %ROI #s to plot
else
    tunedList = sort([], 'ascend'); % If manually entering ROIs, put list in here to plot
end

basedir = '/Volumes/Projects/2P5XFAD/JarascopeData/'; % full data directory path to build subsequent filepaths from
filepathparts = strsplit(plotDir, '/'); 
mouseID = filepathparts{6}; 
sessionID = filepathparts{end};
sessionParts = strsplit(sessionID, '-');
sessionDate = strcat(sessionParts{1}, '-', sessionParts{2}, '-', sessionParts{3});
figdir = '/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Figs';

matOps = dir(fullfile(plotDir,'wehr*.mat'));
if isempty(matOps)
    track2pLog = 1;
    originalMatOps = dir(fullfile(basedir, mouseID, sessionID,'*.mat'));
    load(fullfile(originalMatOps.folder, originalMatOps.name))
else
    track2pLog = 0;
    load(fullfile(matOps.folder, matOps.name));
end

if track2pLog == 1
    cd(fullfile(plotDir, '/suite2p/plane0'))
    F = readNPY('F.npy');
    Fneu = readNPY('Fneu.npy');
    iscell = readNPY('iscell.npy');
    iscellList = iscell; clear iscell
    load('stat.mat')
else
    load(fullfile(plotDir, 'suite2p/plane0/Fall.mat'));
    iscellList = load(fullfile(plotDir,'suite2p/plane0/Fall.mat'), 'iscell');
    iscellList = iscellList.iscell;
    clear iscell
end

iscellLog = logical(iscellList(:, 1));
cellsToPlot = F(iscellLog, :);
neucellsToPlot = Fneu(iscellLog, :);

corrScalar = 0.7;
cellsToPlotCorr = cellsToPlot - (neucellsToPlot * corrScalar);
cellsMean = mean(cellsToPlot, 2);

h5Ops = dir(fullfile(plotDir,'wehr*.h5'));
if isempty(h5Ops)
    h5Ops = dir(fullfile(basedir, mouseID, sessionID,'*.h5'));
end
h5path = fullfile(h5Ops.folder, h5Ops.name);

tones = h5read(h5path, '/resultsData/currentFreq');
intensities = h5read(h5path, '/resultsData/currentIntensity');
allTones = unique(tones);
allInts = unique(intensities); allInts = flip(allInts);
frames = info.frame;
if rem(length(info.frame), length(tones)) == 2
elseif rem(length(info.frame), length(tones)) == (length(tones) - 1)
elseif ~(length(frames)/2 == length(tones))
    frames = frames(1:(end-2));
end
frameIndex = 1:2:length(frames);
frames = frames(frameIndex);

nCond = 0;
for iFreq = 1:length(allTones)
    for iInt = 1:length(allInts)
        nCond = nCond + 1;
        tempTimestamps = frames(tones == allTones(iFreq));
        tempTimestampsInt = frames(intensities == allInts(iInt));
        timestamps{iFreq, iInt} = tempTimestamps(ismember(tempTimestamps, tempTimestampsInt));
        nReps(nCond) = length(timestamps{iFreq, iInt});
    end
end
minReps = min(nReps);
goodStats = stat(iscellLog);
cmap = parula(length(allTones)-1);
cmap = [0,0,0; cmap];

for iROI = 1:length(tunedList)
    if iROI == 1
        currStats = goodStats(tunedList);
        subplot1(2,1, 'Min', [0.05, 0.05], 'Gap', [0.01, 0.01]);
        fig = gcf; orient(fig, 'portrait');
    end
    
    for iTone = 1:length(timestamps)
        for iInt = 1:length(allInts)
            currTimestamps = timestamps{iTone, iInt};
            if length(currTimestamps) > minReps
                currTimestamps = currTimestamps(1:minReps);
            end
            for iTrial = 1:length(currTimestamps)
                currRange = (currTimestamps(iTrial) - 10):(currTimestamps(iTrial) + 20);
                normRange = (currTimestamps(iTrial) - 11):(currTimestamps(iTrial) -1);
                if ~isempty(currRange(currRange <= 0))
                    currRangeLog = currRange < 1;
                    currRange(currRangeLog) = 1;
                end
                if ~isempty(currRange(currRange > frames(end)))
                    currRangeLog = currRange > size(cellsToPlotCorr, 2);
                    currRange(currRangeLog) = size(cellsToPlotCorr, 2);
                end
                if sum(normRange <= 0) == length(normRange) || sum(normRange <= 0) >= 1
                    normRange = (currTimestamps(iTrial) + 21):(currTimestamps(iTrial) + 30);
                end
%                 currTrace = (cellsToPlotCorr(tunedList(iROI), currRange) - mean(cellsToPlotCorr(tunedList(iROI), normRange)))/mean(cellsToPlotCorr(tunedList(iROI), normRange));
                currTrace = (cellsToPlotCorr(tunedList(iROI), currRange) - mean(cellsToPlotCorr(tunedList(iROI), :)))/mean(cellsToPlotCorr(tunedList(iROI), :));
                meanRange(iTrial, :) = currTrace;
            end
            meanRanges{iTone, iInt} = meanRange';
        end
    end
    nCond = 0;
    for iFreq = 1:length(allTones)
        for iDB = 1:length(allInts)
            nCond = nCond + 1;
            meanTrace{iFreq, iDB} = mean(meanRanges{iFreq, iDB}, 2);
            peakvals(iFreq, iDB) = max(meanTrace{iFreq, iDB});
        end
    end
    [tempPeakResponse, bestTones] = max(peakvals, [], [1]);
    [meanPeakResponse(iROI), bestInt] = max(tempPeakResponse);
    peakval = bestTones(bestInt);
%     if tunedList(iROI) == 5 % Override code to hardcode in best frequency, used if noise from another frequency eclipses tone response. Fixed by taking Fnull over enter session SFM 3/24/25
%         peakval = 3;
%     end
    
    currStats{1, iROI}.ypix = abs(currStats{1, iROI}.ypix - 512);
    subplot1(1);
    scats1 = scatter(mean(currStats{1, iROI}.xpix), mean(currStats{1, iROI}.ypix), 60, 'o', 'LineWidth', 1, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', cmap(peakval,:));
    text((mean(currStats{1, iROI}.xpix) + 5), (mean(currStats{1, iROI}.ypix) - 5), num2str(tunedList(iROI)), 'FontSize', 8, 'HorizontalAlignment', 'left');
    
    subplot1(2);
    scats2(iROI) = scatter(mean(currStats{1, iROI}.xpix), mean(currStats{1, iROI}.ypix), 60, 'o', 'LineWidth', 1, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', cmap(peakval,:));
    text((mean(currStats{1, iROI}.xpix) + 5), (mean(currStats{1, iROI}.ypix) - 5), num2str(tunedList(iROI)), 'FontSize', 8, 'HorizontalAlignment', 'left');
    
    scat(iROI) = scats1(1);
    scatFreq(iROI) = peakval;
    hold on
    subplot1(1);
    hold on
end
allbesttones = unique(scatFreq);
subplot1(1); xlim([0 512]); ylim([0 512]);
title(sprintf('Tonotopic Plot of Mean Image %s %s', mouseID, sessionID), 'Position', [256, 525]);
text(256, 520, sprintf('%G um, %G um, %G um from Brightfield', info.config.knobby.pos.x, info.config.knobby.pos.y, info.config.knobby.pos.z), 'HorizontalAlignment', 'center');
subplot1(2); xlim([0 512]); ylim([0 512]);
subplot1(1);
for i = 1:length(allTones)
    [~, col] = find(scatFreq == i);
    if isempty(col)
        legendMap(i) = 0;
    else
        legendMap(i) = scat(col(1));
    end
end
legendlabels = {'WN', '2 kHz', '4 kHz', '8 kHz', '16 kHz', '32 kHz'};
if length(allbesttones) == 6
    lgd = legend(legendMap, legendlabels, 'Location', 'northwest');
% else
%     lgd = imread('/Users/sammehan/Documents/Wehr Lab/Alzheimers2P/Misc./tonotopylegend.png');
%     imshow(lgd);
end
subplot1(2);
for iAlpha = 1:length(tunedList)
    normMeanPeakResponse(iAlpha) = (meanPeakResponse(iAlpha))/(max(meanPeakResponse));
    normMeanPeakResponse(iAlpha) = sqrt(normMeanPeakResponse(iAlpha));
end
for iCell = 1:length(tunedList)
    scats2(iCell).MarkerFaceAlpha = normMeanPeakResponse(iCell);
end
%savenamePS = fullfile(figdir, sprintf('TonotopyPlot-%s-%s.ps', mouseID, sessionDate));
savenamePDF = fullfile(figdir, sprintf('TonotopyPlot-%s-%s.pdf', mouseID, sessionDate));
if ~exist(savenamePDF)
    exportgraphics(gcf, savename);
else
    exportgraphics(gcf, savename, 'Append', true);
end
% print(savenamePS, '-dpsc2', '-append', '-fillpage');
hold off
close all